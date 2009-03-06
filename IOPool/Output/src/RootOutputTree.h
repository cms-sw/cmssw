#ifndef IOPool_Output_RootOutputTree_h
#define IOPool_Output_RootOutputTree_h

/*----------------------------------------------------------------------

RootOutputTree.h // used by ROOT output modules

----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/BranchType.h"

#include "TTree.h"

class TFile;
class TBranch;

namespace edm {

  class RootOutputTree : private boost::noncopyable {
  public:
    // Constructor for trees with no fast cloning
    template <typename T>
    RootOutputTree(T* , // first argument is a dummy so that the compiiler can resolve the match.
		   boost::shared_ptr<TFile> filePtr,
		   BranchType const& branchType,
		   typename T::Auxiliary const*& pAux,
		   typename T::EntryInfoVector *& pEntryInfoVector,
		   int bufSize,
		   int splitLevel,
                   int treeMaxVirtualSize) :
      filePtr_(filePtr),
      tree_(makeTTree(filePtr.get(), BranchTypeToProductTreeName(branchType), splitLevel)),
      metaTree_(makeTTree(filePtr.get(), BranchTypeToMetaDataTreeName(branchType), 0)),
      auxBranch_(0),
      producedBranches_(),
      metaBranches_(),
      readBranches_(),
      unclonedReadBranches_(),
      unclonedReadBranchNames_(),
      currentlyFastCloning_(),
      basketSize_(bufSize),
      splitLevel_(splitLevel) {

      if (treeMaxVirtualSize >= 0) tree_->SetMaxVirtualSize(treeMaxVirtualSize);
      auxBranch_ = tree_->Branch(BranchTypeToAuxiliaryBranchName(branchType).c_str(), &pAux, bufSize, 0);
      readBranches_.push_back(auxBranch_);  

      branchEntryInfoBranch_ = metaTree_->Branch(BranchTypeToBranchEntryInfoBranchName(branchType).c_str(),
                                                 &pEntryInfoVector, bufSize, 0);
      metaBranches_.push_back(branchEntryInfoBranch_);
  }

    ~RootOutputTree() {}
    
    static void fastCloneTTree(TTree *in, TTree *out);

    static TTree * makeTTree(TFile *filePtr, std::string const& name, int splitLevel);

    static TTree * assignTTree(TFile *file, TTree * tree);

    static void writeTTree(TTree *tree);

    bool isValid() const;

    void addBranch(BranchDescription const& prod,
		   void const*& pProd, bool produced);

    bool checkSplitLevelAndBasketSize(TTree *inputTree) const;

    void fastCloneTree(TTree *tree);

    void fillTree() const;

    void writeTree() const;

    TTree *const tree() const {
      return tree_;
    }

    TTree *const metaTree() const {
      return metaTree_;
    }

    void setEntries() {
      if (tree_->GetNbranches() != 0) tree_->SetEntries(-1);
      if (metaTree_->GetNbranches() != 0) metaTree_->SetEntries(-1);
    }

    void beginInputFile(bool fastCloning) {
      currentlyFastCloning_ = fastCloning;
    }

    bool
    uncloned(std::string const& branchName) const {
	return unclonedReadBranchNames_.find(branchName) != unclonedReadBranchNames_.end();
    }

  private:
    static void fillTTree(TTree *tree, std::vector<TBranch *> const& branches);
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using smart pointers here will do no good.
    boost::shared_ptr<TFile> filePtr_;
    TTree *const tree_;
    TTree *const metaTree_;
    TBranch * auxBranch_;
    TBranch * branchEntryInfoBranch_;
    std::vector<TBranch *> producedBranches_; // does not include cloned branches
    std::vector<TBranch *> metaBranches_;
    std::vector<TBranch *> readBranches_;
    std::vector<TBranch *> unclonedReadBranches_;
    std::set<std::string> unclonedReadBranchNames_;
    bool currentlyFastCloning_;
    int basketSize_;
    int splitLevel_;
  };
}
#endif
