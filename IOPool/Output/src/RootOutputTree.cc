
#include "RootOutputTree.h"

#include "TFile.h"
#include "TBranch.h"
#include "TTreeCloner.h"
#include "Rtypes.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "boost/bind.hpp"
#include <limits>

namespace edm {
  TTree *
  RootOutputTree::assignTTree(TFile * filePtr, TTree * tree) {
    tree->SetDirectory(filePtr);
    // Turn off autosaving because it is such a memory hog and we are not using
    // this check-pointing feature anyway.
    tree->SetAutoSave(std::numeric_limits<Long64_t>::max());
    return tree;
  }

  TTree *
  RootOutputTree::makeTTree(TFile * filePtr, std::string const& name, int splitLevel) {
    TTree *tree = new TTree(name.c_str(), "", splitLevel);
    if (!tree) throw edm::Exception(edm::errors::FatalRootError)
      << "Failed to create the tree: " << name << "\n";
    if (tree->IsZombie())
      throw edm::Exception(edm::errors::FatalRootError)
	<< "Tree: " << name << " is a zombie." << "\n";
				    
    return assignTTree(filePtr, tree);
  }

  bool RootOutputTree::checkSplitLevelAndBasketSize(TTree *inputTree) const {

    if (inputTree == 0) return false;

    // Do the split level and basket size match in the input and output?
    for (std::vector<TBranch *>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end();
      it != itEnd; ++it) {

      TBranch* outputBranch = *it;
      if (outputBranch != 0) {
        TBranch* inputBranch = inputTree->GetBranch(outputBranch->GetName());

        if (inputBranch != 0) {
          if (inputBranch->GetSplitLevel() != outputBranch->GetSplitLevel() ||
              inputBranch->GetBasketSize() != outputBranch->GetBasketSize()) {
            LogInfo("FastCloning")
              << "Fast Cloning disabled because split level or basket size do not match";
            return false;
          }
        }
      }
    }
    return true;
  }


  void
  RootOutputTree::fastCloneTTree(TTree *in, TTree *out) {
    if (in->GetEntries() != 0) {
      TTreeCloner cloner(in, out, "");
      if (!cloner.IsValid()) {
        throw edm::Exception(edm::errors::FatalRootError)
          << "invalid TTreeCloner\n";
      }
      out->SetEntries(out->GetEntries() + in->GetEntries());
      cloner.Exec();
    }
  }
 
  void
  RootOutputTree::writeTTree(TTree *tree) {
    if (tree->GetNbranches() != 0) {
      tree->SetEntries(-1);
    }
    tree->AutoSave();
  }

  void
  RootOutputTree::fillTTree(TTree * tree, std::vector<TBranch *> const& branches) {
    for_all(branches, boost::bind(&TBranch::Fill, _1));
  }

  void
  RootOutputTree::writeTree() const {
    writeTTree(tree_);
    writeTTree(metaTree_);
  }

  void
  RootOutputTree::fastCloneTree(TTree *tree) {
    unclonedReadBranches_.clear();
    unclonedReadBranchNames_.clear();
    if (currentlyFastCloning_) {
      fastCloneTTree(tree, tree_);
      for (std::vector<TBranch *>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end();
	  it != itEnd; ++it) {
	if ((*it)->GetEntries() != tree_->GetEntries()) {
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
    if (currentlyFastCloning_) {
      fillTTree(tree_, unclonedReadBranches_);
    } else {
      fillTTree(tree_, readBranches_);
    }
  }

  void
  RootOutputTree::addBranch(BranchDescription const& prod,
			    void const*& pProd, bool produced) {
      prod.init();
      TBranch *branch = tree_->Branch(prod.branchName().c_str(),
		 prod.wrappedName().c_str(),
		 &pProd,
		 (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize()),
		 (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel()));
      if (produced) {
	producedBranches_.push_back(branch);
      } else {
        readBranches_.push_back(branch);
      }
  }
}
