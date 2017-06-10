#ifndef Alignment_HIPAlignmentAlgorithm_HIPMonitorConfig_h
#define Alignment_HIPAlignmentAlgorithm_HIPMonitorConfig_h

#include <vector>
#include <string>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

struct HIPMonitorConfig{
  const edm::ParameterSet cfgMonitor;

  const std::string outfilecore;

  const bool fillTrackMonitoring;
  const int maxEventsPerJob;

  const bool fillTrackHitMonitoring;
  const int maxHits;

  std::string outfile;

  int eventCounter;
  int hitCounter;

  HIPMonitorConfig(const edm::ParameterSet& cfg);
  HIPMonitorConfig(const HIPMonitorConfig& other);
  ~HIPMonitorConfig(){}

  bool checkNevents();
  bool checkNhits();
};

#endif
