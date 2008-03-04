#ifndef IOPool_Output_RootOutputTree_h
#define IOPool_Output_RootOutputTree_h

/*----------------------------------------------------------------------

RootOutputTree.h // used by ROOT output modules

$Id: RootOutputTree.h,v 1.23 2008/02/25 16:47:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/Selections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "TBranch.h"
#include "TTree.h"
class TFile;

namespace edm {

  class RootOutputTree {
  public:
    // Constructor for trees with no fast cloning
    template <typename T>
    RootOutputTree(boost::shared_ptr<TFile> filePtr,
		   BranchType const& branchType,
		   T const*& pAux,
		   ProductStatusVector const*& pProdStats,
		   int bufSize,
		   int splitLevel,
		   bool fastCloning = false,
		   bool fastMetaCloning = false,
		   TTree * tree = 0,
		   TTree * metaTree = 0,
		   Selections const& dropList = Selections(),
		   std::vector<std::string> renamedList = std::vector<std::string>()) :
      filePtr_(filePtr),
      tree_(fastCloning ?
		pseudoCloneTTree(filePtr.get(), tree, dropList, renamedList, splitLevel)
		:
		makeTTree(filePtr.get(), BranchTypeToProductTreeName(branchType), splitLevel)),
      metaTree_(fastMetaCloning ?
		cloneTTree(filePtr.get(), metaTree, Selections(), renamedList)
		:
		makeTTree(filePtr.get(), BranchTypeToMetaDataTreeName(branchType), 0)),
      infoTree_(makeTTree(filePtr.get(), BranchTypeToInfoTreeName(branchType), 0)),
      auxBranch_(0),
      statusBranch_(0),
      branches_(),
      metaBranches_(),
      infoBranches_(),
      clonedBranches_(),
      clonedMetaBranches_(),
      currentlyFastCloning_(),
      currentlyFastMetaCloning_(),
      basketSize_(bufSize),
      splitLevel_(splitLevel),
      branchNames_() {

      auxBranch_ = tree_->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
      if (auxBranch_ != 0) {
	auxBranch_->SetAddress(&pAux);
        clonedBranches_.push_back(auxBranch_);
      } else {
        auxBranch_ = tree_->Branch(BranchTypeToAuxiliaryBranchName(branchType).c_str(), &pAux, bufSize, 0);
        branches_.push_back(auxBranch_);
      }
      statusBranch_ = infoTree_->Branch(BranchTypeToProductStatusBranchName(branchType).c_str(), &pProdStats, bufSize, 0);
      infoBranches_.push_back(statusBranch_);
    }

    static void fastCloneTTree(TTree *in, TTree *out);

    static TTree * cloneTTree(TFile *filePtr, TTree *tree, Selections const& dropList, std::vector<std::string> const& renamedList);

    static TTree * pseudoCloneTTree(TFile *filePtr, TTree *tree, Selections const& dropList, std::vector<std::string> const& renamedList, int splitLevel);

    static TTree * makeTTree(TFile *filePtr, std::string const& name, int splitLevel);

    static TTree * assignTTree(TFile *file, TTree * tree);

    static void writeTTree(TTree *tree);

    static void pruneTTree(TTree *tree, Selections const& dropList, std::vector<std::string> const& renamedList);

    bool isValid() const;

    void addBranch(BranchDescription const& prod, bool selected, EntryDescriptionID*& pEntryDescID, void const*& pProd);

    std::vector<std::string> const& branchNames() const {return branchNames_;}

    void fastCloneTree(TTree *tree, TTree *metaTree);

    void fillTree() const;

    void writeTree() const;

    TTree *const tree() const {
      return tree_;
    }

    TTree *const metaTree() const {
      return metaTree_;
    }

    TTree *const infoTree() const {
      return infoTree_;
    }

    void setEntries() {
      if (tree_->GetNbranches() != 0) tree_->SetEntries(-1);
      if (metaTree_->GetNbranches() != 0) metaTree_->SetEntries(-1);
    }

    void beginInputFile(bool fastCloning, bool fastMetaCloning) {
      currentlyFastCloning_ = fastCloning;
      currentlyFastMetaCloning_ = fastMetaCloning;
    }

  private:
    static void fillTTree(TTree *tree, std::vector<TBranch *> const& branches);
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using smart pointers here will do no good.
    boost::shared_ptr<TFile> filePtr_;
    TTree *const tree_;
    TTree *const metaTree_;
    TTree *const infoTree_;
    TBranch * auxBranch_;
    TBranch * statusBranch_;
    std::vector<TBranch *> branches_; // does not include cloned branches
    std::vector<TBranch *> metaBranches_; // does not include cloned branches
    std::vector<TBranch *> infoBranches_;
    std::vector<TBranch *> clonedBranches_;
    std::vector<TBranch *> clonedMetaBranches_;
    bool currentlyFastCloning_;
    bool currentlyFastMetaCloning_;
    int basketSize_;
    int splitLevel_;
    std::vector<std::string> branchNames_;
  };
}
#endif
