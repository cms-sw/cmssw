#ifndef FWCore_ParameterSet_validateTopLevelParameterSets_h
#define FWCore_ParameterSet_validateTopLevelParameterSets_h

namespace edm {

  class ParameterSet;
  class ParameterSetDescription;

  void validateTopLevelParameterSets(ParameterSet* processParameterSet);
  void fillOptionsDescription(ParameterSetDescription& description);
  void fillMaxEventsDescription(ParameterSetDescription& description);
  void fillMaxLuminosityBlocksDescription(ParameterSetDescription& description);
  void fillMaxSecondsUntilRampdownDescription(ParameterSetDescription& description);
  void dumpOptionsToLogFile(unsigned int nThreads,
                            unsigned int nStreams,
                            unsigned int nConcurrentLumis,
                            unsigned int nConcurrentRuns);
}  // namespace edm
#endif
