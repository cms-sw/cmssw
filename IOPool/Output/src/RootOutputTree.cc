#include "RootOutputTree.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeCloner.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include "boost/bind.hpp"
#include <algorithm>
#include <cstring>

namespace edm {

  TTree *
  RootOutputTree::assignTTree(TFile * filePtr, TTree * tree) {
    tree->SetDirectory(filePtr);
    // Turn off autosaving because it is such a memory hog and we are not using
    // this check-pointing feature anyway.
    tree->SetAutoSave(400000000000LL);
    return tree;
  }

  TTree *
  RootOutputTree::makeTTree(TFile * filePtr, std::string const& name, int splitLevel) {
    TTree *tree = new TTree(name.c_str(), "", splitLevel);
    return assignTTree(filePtr, tree);
  }

  TTree *
  RootOutputTree::pseudoCloneTTree(TFile * filePtr, TTree *tree, int splitLevel) {
    TTree *newTree = new TTree(tree->GetName(), tree->GetTitle(), splitLevel);
    return assignTTree(filePtr, newTree);
  }

  TTree *
  RootOutputTree::cloneTTree(TFile * filePtr, TTree *tree, Selections const& dropList, std::vector<std::string> const& renamedList) {
    pruneTTree(tree, dropList, renamedList);
    TTree *newTree = tree->CloneTree(0);
    tree->SetBranchStatus("*", 1);
//  Break association of the tree with its clone
    tree->GetListOfClones()->Remove(newTree);
    newTree->ResetBranchAddresses();
    return assignTTree(filePtr, newTree);
  }

  void
  RootOutputTree::fastCloneTTree(TTree *in, TTree *out) {
    TTreeCloner cloner(in, out, "");
    if (!cloner.IsValid()) {
       throw edm::Exception(edm::errors::FatalRootError)
         << "invalid TTreeCloner\n";
    }
    out->SetEntries(out->GetEntries() + in->GetEntries());
    cloner.Exec();
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
  RootOutputTree::fastCloneTree(TTree *tree, TTree *metaTree) {
    fastCloneTTree(metaTree, metaTree_);
    fastCloneTTree(tree, tree_);
  }

  void
  RootOutputTree::fillTree() const {
    fillTTree(metaTree_, metaBranches_);
    fillTTree(tree_, branches_);
    if (!currentlyFastCloning_) {
      fillTTree(metaTree_, clonedMetaBranches_);
      fillTTree(tree_, clonedBranches_);
    }
  }

  void
  RootOutputTree::pruneTTree(TTree *tree, Selections const& dropList, std::vector<std::string> const& renamedList) {
    // Since we don't know the history, make sure all branches are activated.
    tree->SetBranchStatus("*", 1);
  
    // Iterate over the list of branch names to drop
    for(Selections::const_iterator it = dropList.begin(), itEnd = dropList.end(); it != itEnd; ++it) {
      if ((*it)->present()) {
        std::string branchName = (*it)->branchName();
        char lastchar = branchName.at(branchName.size()-1);
        if(lastchar == '.') {
          branchName += "*";
        } else {
          branchName += ".*";
        }
        tree->SetBranchStatus(branchName.c_str(), 0);
      }
    }

    for(std::vector<std::string>::const_iterator it = renamedList.begin(), itEnd = renamedList.end(); it != itEnd; ++it) {
      std::string branchName = *it;
      char lastchar = branchName.at(branchName.size()-1);
      if(lastchar == '.') {
        branchName += "*";
      } else {
        branchName += ".*";
      }
      tree->SetBranchStatus(branchName.c_str(), 0);
    }
  }

  void
  RootOutputTree::addBranch(BranchDescription const& prod, bool selected, BranchEntryDescription const*& pProv, void const*& pProd) {
      prod.init();
      TBranch * meta = metaTree_->GetBranch(prod.branchName().c_str());
      if (meta != 0) {
	meta->SetAddress(&pProv);
        clonedMetaBranches_.push_back(meta);
      } else {
        meta = metaTree_->Branch(prod.branchName().c_str(), &pProv, basketSize_, 0);
        metaBranches_.push_back(meta);
      }
      if (selected) {
        TBranch * oldBranch = (fastCloning_ ? oldTree_->GetBranch(prod.branchName().c_str()) : 0);
        if (oldBranch != 0) {
	  TBranch *branch = tree_->Branch(prod.branchName().c_str(),
		   prod.wrappedName().c_str(),
		   &pProd,
		   (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize()),
		   oldBranch->GetSplitLevel());
          clonedBranches_.push_back(branch);
	} else {
	  TBranch *branch = tree_->Branch(prod.branchName().c_str(),
		   prod.wrappedName().c_str(),
		   &pProd,
		   (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize()),
		   (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel()));
          branches_.push_back(branch);
        }
	// we want the new branch name for the JobReport
	branchNames_.push_back(prod.branchName());
      }
  }
}
