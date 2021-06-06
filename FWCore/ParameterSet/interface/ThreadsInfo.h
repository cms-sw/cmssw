#ifndef FWCore_ParameterSet_ThreadsInfo_h
#define FWCore_ParameterSet_ThreadsInfo_h

namespace edm {

  class ParameterSet;
  class ParameterSetDescription;

  constexpr unsigned int s_defaultNumberOfThreads = 1;
  constexpr unsigned int s_defaultSizeOfStackForThreadsInKB = 10 * 1024;  //10MB

  struct ThreadsInfo {
    unsigned int nThreads_ = s_defaultNumberOfThreads;
    unsigned int stackSize_ = s_defaultSizeOfStackForThreadsInKB;
  };

  ThreadsInfo threadOptions(edm::ParameterSet const& pset);
  void setThreadOptions(ThreadsInfo const& threadsInfo, edm::ParameterSet& pset);

}  // namespace edm
#endif
