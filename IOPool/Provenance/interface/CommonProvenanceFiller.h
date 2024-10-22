class TTree;

namespace edm {

  class ProcessHistoryRegistry;

  void fillParameterSetBranch(TTree* parameterSetsTree, int basketSize);

  void fillProcessHistoryBranch(TTree* metaDataTree,
                                int basketSize,
                                ProcessHistoryRegistry const& processHistoryRegistry);
}  // namespace edm
