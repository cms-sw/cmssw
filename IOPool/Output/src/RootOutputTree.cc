
#include "RootOutputTree.h"

#include "TFile.h"
#include "TBranch.h"
#include "TBranchElement.h"
#include "TTreeCloner.h"
#include "TCollection.h"
#include "Rtypes.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/bind.hpp"
#include <limits>

namespace edm {
  TTree*
  RootOutputTree::assignTTree(TFile* filePtr, TTree* tree) {
    tree->SetDirectory(filePtr);
    // Turn off autosaving because it is such a memory hog and we are not using
    // this check-pointing feature anyway.
    tree->SetAutoSave(std::numeric_limits<Long64_t>::max());
    return tree;
  }

  TTree*
  RootOutputTree::makeTTree(TFile* filePtr, std::string const& name, int splitLevel) {
    TTree* tree = new TTree(name.c_str(), "", splitLevel);
    if(!tree) throw edm::Exception(errors::FatalRootError)
      << "Failed to create the tree: " << name << "\n";
    if(tree->IsZombie())
      throw edm::Exception(errors::FatalRootError)
	<< "Tree: " << name << " is a zombie." << "\n";
				    
    return assignTTree(filePtr, tree);
  }

  bool
  RootOutputTree::checkSplitLevelsAndBasketSizes(TTree* inputTree) const {

    assert (inputTree != 0);

    // Do the split level and basket size match in the input and output?
    for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end();
      it != itEnd; ++it) {

      TBranch* outputBranch = *it;
      if(outputBranch != 0) {
        TBranch* inputBranch = inputTree->GetBranch(outputBranch->GetName());

        if(inputBranch != 0) {
          if(inputBranch->GetSplitLevel() != outputBranch->GetSplitLevel() ||
              inputBranch->GetBasketSize() != outputBranch->GetBasketSize()) {
            return false;
          }
        }
      }
    }
    return true;
  }

  namespace {
    bool checkMatchingBranches(TBranchElement* inputBranch, TBranchElement* outputBranch) {
      if(inputBranch->GetStreamerType() != outputBranch->GetStreamerType()) {
        return false;
      }
      TObjArray* inputArray = inputBranch->GetListOfBranches();
      TObjArray* outputArray = outputBranch->GetListOfBranches();

      if(outputArray->GetSize() < inputArray->GetSize()) {
        return false;
      }
      TIter iter(outputArray);
      TObject* obj = 0;
      while(obj = iter.Next()) {
        TBranchElement* outBranch = dynamic_cast<TBranchElement*>(obj);
        if(outBranch) {
	  TBranchElement* inBranch = dynamic_cast<TBranchElement*>(inputBranch->FindBranch(outBranch->GetName()));
	  if(!inBranch) {
	    return false;
	  }
	  if(!checkMatchingBranches(inBranch, outBranch)) {
	    return false;
	  }
        }
      }
      return true;
    }
  }

  bool RootOutputTree::checkIfFastClonable(TTree* inputTree) const {

    if(inputTree == 0) return false;

    // Do the sub-branches match in the input and output.  Extra sub-branches in the input are OK for fast cloning, but not in the output.
    for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end(); it != itEnd; ++it) {
      TBranchElement* outputBranch = dynamic_cast<TBranchElement*>(*it);
      if(outputBranch != 0) {
	TBranchElement* inputBranch = dynamic_cast<TBranchElement*>(inputTree->GetBranch(outputBranch->GetName()));
        if(inputBranch != 0) {
	  // We have a matching top level branch. Do the recursive check on subbranches.
	  if(!checkMatchingBranches(inputBranch, outputBranch)) {
            LogInfo("FastCloning")
              << "Fast Cloning disabled because a data member has been added to  split branch: " << inputBranch->GetName() << "\n.";
	    return false;
	  }
	}
      }
    }
    return true;
  }

  void
  RootOutputTree::fastCloneTTree(TTree* in, TTree* out) {
    if(in->GetEntries() != 0) {
      TTreeCloner cloner(in, out, "");
      if(!cloner.IsValid()) {
        throw edm::Exception(errors::FatalRootError)
          << "invalid TTreeCloner\n";
      }
      out->SetEntries(out->GetEntries() + in->GetEntries());
      cloner.Exec();
    }
  }
 
  void
  RootOutputTree::writeTTree(TTree* tree) {
    if(tree->GetNbranches() != 0) {
      tree->SetEntries(-1);
    }
    setRefCoreStreamer(true);
    tree->AutoSave();
  }

  void
  RootOutputTree::fillTTree(TTree* tree, std::vector<TBranch*> const& branches) {
    for_all(branches, boost::bind(&TBranch::Fill, _1));
  }

  void
  RootOutputTree::writeTree() const {
    writeTTree(tree_);
    writeTTree(metaTree_);
  }

  void
  RootOutputTree::maybeFastCloneTree(TTree* tree) {
    unclonedReadBranches_.clear();
    unclonedReadBranchNames_.clear();
    if(currentlyFastCloning_) {
      fastCloneTTree(tree, tree_);
      for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end();
	  it != itEnd; ++it) {
	if((*it)->GetEntries() != tree_->GetEntries()) {
	  unclonedReadBranches_.push_back(*it);
	  unclonedReadBranchNames_.insert(std::string((*it)->GetName()));
	}
      }
    }
  }

  void
  RootOutputTree::fillTree() const {
    fillTTree(metaTree_, metaBranches_);
    fillTTree(tree_, producedBranches_);
    if(currentlyFastCloning_) {
      fillTTree(tree_, unclonedReadBranches_);
    } else {
      fillTTree(tree_, readBranches_);
    }
  }

  void
  RootOutputTree::addBranch(BranchDescription const& prod,
			    void const*& pProd,
			    int splitLevel,
			    int basketSize,
			    bool produced) {
      prod.init();
      assert(splitLevel != BranchDescription::invalidSplitLevel);
      assert(basketSize != BranchDescription::invalidBasketSize);
      TBranch* branch = tree_->Branch(prod.branchName().c_str(),
		 prod.wrappedName().c_str(),
		 &pProd,
		 basketSize,
		 splitLevel);
      if(produced) {
	producedBranches_.push_back(branch);
      } else {
        readBranches_.push_back(branch);
      }
  }

  void
  RootOutputTree::close() {
    // The TFile was just closed.
    // Just to play it safe, zero all pointers to quantities in the file.
    auxBranch_ = branchEntryInfoBranch_ = 0;
    producedBranches_.clear();
    metaBranches_.clear();
    readBranches_.clear();
    unclonedReadBranches_.clear();
    tree_ = metaTree_ = 0;
    filePtr_.reset();
  }
}
