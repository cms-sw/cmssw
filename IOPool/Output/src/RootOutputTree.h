#ifndef IOPool_Output_RootOutputTree_h
#define IOPool_Output_RootOutputTree_h

/*----------------------------------------------------------------------

RootOutputTree.h // used by ROOT output modules

----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include <memory>

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include "TTree.h"

class TFile;
class TBranch;

namespace edm {
  class RootOutputTree {
  public:
    RootOutputTree(std::shared_ptr<TFile> filePtr,
                   BranchType const& branchType,
                   int splitLevel,
                   int treeMaxVirtualSize);

    ~RootOutputTree() {}

    RootOutputTree(RootOutputTree const&) = delete; // Disallow copying and moving
    RootOutputTree& operator=(RootOutputTree const&) = delete; // Disallow copying and moving

    template <typename T>
    void
    addAuxiliary(std::string const& branchName, T const*& pAux, int bufSize, bool allowCloning=true) {
      if(allowCloning) {
        auxBranches_.push_back(tree_->Branch(branchName.c_str(), &pAux, bufSize, 0));
      } else {
        unclonedAuxBranches_.push_back(tree_->Branch(branchName.c_str(), &pAux, bufSize, 0));    
      }
    }

    template <typename T>
    void
    addAuxiliary(std::string const& branchName, T*& pAux, int bufSize,bool allowCloning=true) {
      if(allowCloning) {
        auxBranches_.push_back(tree_->Branch(branchName.c_str(), &pAux, bufSize, 0));
      } else {
        unclonedAuxBranches_.push_back(tree_->Branch(branchName.c_str(), &pAux, bufSize, 0));    
      }
    }

    void fastCloneTTree(TTree* in, std::string const& option);

    static TTree* makeTTree(TFile* filePtr, std::string const& name, int splitLevel);

    static TTree* assignTTree(TFile* file, TTree* tree);

    static void writeTTree(TTree* tree);

    bool isValid() const;

    void addBranch(std::string const& branchName,
                   std::string const& className,
                   void const*& pProd,
                   int splitLevel,
                   int basketSize,
                   bool produced);

    bool checkSplitLevelsAndBasketSizes(TTree* inputTree) const;

    bool checkIfFastClonable(TTree* inputTree) const;

    bool checkEntriesInReadBranches(Long64_t expectedNumberOfEntries) const;

    void maybeFastCloneTree(bool canFastClone, bool canFastCloneAux, TTree* tree, std::string const& option);

    void fillTree() const;

    void writeTree() const;

    TTree* tree() const {
      return tree_;
    }

    void setEntries() {
      if(tree_->GetNbranches() != 0) tree_->SetEntries(-1);
    }

    bool
    uncloned(std::string const& branchName) const {
      return clonedReadBranchNames_.find(branchName) == clonedReadBranchNames_.end();
    }

    void close();

    void optimizeBaskets(ULong64_t size) {
      tree_->OptimizeBaskets(size);
    }

    void setAutoFlush(Long64_t size) {
      tree_->SetAutoFlush(size);
    }
  private:
    static void fillTTree(std::vector<TBranch*> const& branches);
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore, using smart pointers here will do no good.
    std::shared_ptr<TFile> filePtr_;
    TTree* tree_;
    std::vector<TBranch*> producedBranches_; // does not include cloned branches
    std::vector<TBranch*> readBranches_;
    std::vector<TBranch*> auxBranches_;
    std::vector<TBranch*> unclonedAuxBranches_;
    std::vector<TBranch*> unclonedReadBranches_;
    std::set<std::string> clonedReadBranchNames_;
    bool currentlyFastCloning_;
    bool fastCloneAuxBranches_;
  };
}
#endif
