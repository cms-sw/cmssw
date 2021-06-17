#include "FWCore/Framework/interface/FileBlock.h"

#include <algorithm>

namespace edm {
  void FileBlock::updateTTreePointers(TTree* ev,
                                      TTree* meta,
                                      TTree* lumi,
                                      TTree* lumiMeta,
                                      TTree* run,
                                      TTree* runMeta,
                                      std::vector<TTree*> processBlockTrees,
                                      std::vector<std::string> processesWithProcessBlockTrees) {
    tree_ = ev;
    metaTree_ = meta;
    lumiTree_ = lumi;
    lumiMetaTree_ = lumiMeta;
    runTree_ = run;
    runMetaTree_ = runMeta;
    processBlockTrees_ = std::move(processBlockTrees);
    processesWithProcessBlockTrees_ = std::move(processesWithProcessBlockTrees);
  }

  TTree* FileBlock::processBlockTree(std::string const& processName) const {
    auto it = std::find(processesWithProcessBlockTrees_.begin(), processesWithProcessBlockTrees_.end(), processName);
    if (it != processesWithProcessBlockTrees_.end()) {
      auto index = std::distance(processesWithProcessBlockTrees_.begin(), it);
      return processBlockTrees_[index];
    }
    return nullptr;
  }

  void FileBlock::close() {
    runMetaTree_ = lumiMetaTree_ = metaTree_ = runTree_ = lumiTree_ = tree_ = nullptr;
    std::fill(processBlockTrees_.begin(), processBlockTrees_.end(), nullptr);
  }

}  // namespace edm
