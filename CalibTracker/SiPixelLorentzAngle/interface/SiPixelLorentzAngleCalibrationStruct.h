#ifndef CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngleCalibrationStruct_h
#define CalibTracker_SiPixelLorentzAngle_SiPixelLorentzAngleCalibrationStruct_h

#include "DQMServices/Core/interface/DQMStore.h"
#include <unordered_map>

struct SiPixelLorentzAngleCalibrationHistograms {
public:
  SiPixelLorentzAngleCalibrationHistograms() = default;

  using MonitorMap = std::unordered_map<uint32_t, dqm::reco::MonitorElement*>;

  int nlay;
  std::vector<int> nModules_;
  std::vector<std::string> BPixnewmodulename_;
  std::vector<int> BPixnewDetIds_;
  std::vector<int> BPixnewModule_;
  std::vector<int> BPixnewLayer_;

  std::vector<std::string> FPixnewmodulename_;
  std::vector<int> FPixnewDetIds_;
  std::vector<int> FPixnewDisk_;
  std::vector<int> FPixnewBlade_;
  std::unordered_map<uint32_t, std::vector<uint32_t> > detIdsList;

  MonitorMap h_drift_depth_adc_;
  MonitorMap h_drift_depth_adc2_;
  MonitorMap h_drift_depth_noadc_;
  MonitorMap h_drift_depth_;
  MonitorMap h_mean_;

  dqm::reco::MonitorElement* h_tracks_;
};

#endif
