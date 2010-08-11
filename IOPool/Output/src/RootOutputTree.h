#ifndef IOPool_Output_RootOutputTree_h
#define IOPool_Output_RootOutputTree_h

/*----------------------------------------------------------------------

RootOutputTree.h // used by ROOT output modules

----------------------------------------------------------------------*/

#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/BranchType.h"

#include "TTree.h"

class TFile;
class TBranch;

namespace edm {

  class RootOutputTree : private boost::noncopyable {
  public:
    RootOutputTree(boost::shared_ptr<TFile> filePtr,
		   BranchType const& branchType,
		   std::vector<ProductProvenance>*& pEntryInfoVector,
		   int bufSize,
		   int splitLevel,
                   int treeMaxVirtualSize) :
      filePtr_(filePtr),
      tree_(makeTTree(filePtr.get(), BranchTypeToProductTreeName(branchType), splitLevel)),
      metaTree_(makeTTree(filePtr.get(), BranchTypeToMetaDataTreeName(branchType), 0)),
      producedBranches_(),
      readBranches_(),
      auxBranch_(),
      unclonedReadBranches_(),
      unclonedReadBranchNames_(),
      currentlyFastCloning_() {
      if(treeMaxVirtualSize >= 0) tree_->SetMaxVirtualSize(treeMaxVirtualSize);
      branchEntryInfoBranch_ = metaTree_->Branch(BranchTypeToBranchEntryInfoBranchName(branchType).c_str(),
                                                 &pEntryInfoVector, bufSize, 0);
    }

    ~RootOutputTree() {}

    template <typename T>
    void
    addAuxiliary(BranchType const& branchType, T const*& pAux, int bufSize) {
      auxBranch_ = tree_->Branch(BranchTypeToAuxiliaryBranchName(branchType).c_str(), &pAux, bufSize, 0);
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

    void maybeFastCloneTree(bool canFastClone, TTree* tree, std::string const& option);

    void fillTree() const;

    void writeTree() const;

    TTree* const tree() const {
      return tree_;
    }

    TTree* const metaTree() const {
      return metaTree_;
    }

    void setEntries() {
      if(tree_->GetNbranches() != 0) tree_->SetEntries(-1);
      if(metaTree_->GetNbranches() != 0) metaTree_->SetEntries(-1);
    }

    bool
    uncloned(std::string const& branchName) const {
	return unclonedReadBranchNames_.find(branchName) != unclonedReadBranchNames_.end();
    }

    void close();

  private:
    static void fillTTree(TTree* tree, std::vector<TBranch*> const& branches);
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore, using smart pointers here will do no good.
    boost::shared_ptr<TFile> filePtr_;
    TTree* tree_;
    TTree* metaTree_;
    TBranch* branchEntryInfoBranch_;
    std::vector<TBranch*> producedBranches_; // does not include cloned branches
    std::vector<TBranch*> readBranches_;
    TBranch* auxBranch_;
    std::vector<TBranch*> unclonedReadBranches_;
    std::set<std::string> unclonedReadBranchNames_;
    bool currentlyFastCloning_;
  };
}
#endif
