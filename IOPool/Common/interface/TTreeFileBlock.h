#ifndef IOPool_Common_TTreeFileBlock_h
#define IOPool_Common_TTreeFileBlock_h

#include "FWCore/Framework/interface/FileBlock.h"

#include <string>
#include <utility>
#include <vector>

class TTree;

namespace edm {
  // TTrees are are not owned.
  // Process names and Ttrees must have equal lengths and corresponding order.
  class TTreeFileBlock : public FileBlock {
  public:
    TTreeFileBlock() = default;

    TTreeFileBlock(FileFormatVersion const& version,
                   TTree* ev,
                   TTree* meta,
                   TTree* lumi,
                   TTree* lumiMeta,
                   TTree* run,
                   TTree* runMeta,
                   std::vector<TTree*> processBlockTrees,
                   std::vector<std::string> processesWithProcessBlockTrees,
                   int whyNotFastClonable,
                   std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch,
                   std::string const& fileName,
                   bool branchListIndexesUnchanged,
                   bool modifiedIDs,
                   std::shared_ptr<ProductDependencies const> productDependencies)
        : FileBlock(version,
                    whyNotFastClonable,
                    hasNewlyDroppedBranch,
                    fileName,
                    branchListIndexesUnchanged,
                    modifiedIDs,
                    std::move(productDependencies)),
          tree_(ev),
          metaTree_(meta),
          lumiTree_(lumi),
          lumiMetaTree_(lumiMeta),
          runTree_(run),
          runMetaTree_(runMeta),
          processBlockTrees_(std::move(processBlockTrees)),
          processesWithProcessBlockTrees_(std::move(processesWithProcessBlockTrees)) {}

    void updateTTreePointers(TTree* ev,
                             TTree* meta,
                             TTree* lumi,
                             TTree* lumiMeta,
                             TTree* run,
                             TTree* runMeta,
                             std::vector<TTree*> processBlockTrees,
                             std::vector<std::string> processesWithProcessBlockTrees);

    TTree* tree() const { return tree_; }
    TTree* metaTree() const { return metaTree_; }
    TTree* lumiTree() const { return lumiTree_; }
    TTree* lumiMetaTree() const { return lumiMetaTree_; }
    TTree* runTree() const { return runTree_; }
    TTree* runMetaTree() const { return runMetaTree_; }
    TTree* processBlockTree(std::string const& processName) const;
    std::vector<TTree*> const& processBlockTrees() const { return processBlockTrees_; }
    std::vector<std::string> const& processesWithProcessBlockTrees() const { return processesWithProcessBlockTrees_; }

    void close() override;

  private:
    TTree* tree_ = nullptr;
    TTree* metaTree_ = nullptr;
    TTree* lumiTree_ = nullptr;
    TTree* lumiMetaTree_ = nullptr;
    TTree* runTree_ = nullptr;
    TTree* runMetaTree_ = nullptr;
    std::vector<TTree*> processBlockTrees_;
    std::vector<std::string> processesWithProcessBlockTrees_;
  };
}  // namespace edm
#endif
