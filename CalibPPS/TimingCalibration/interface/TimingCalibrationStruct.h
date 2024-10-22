#ifndef CalibPPS_TimingCalibration_TimingCalibrationStruct_h
#define CalibPPS_TimingCalibration_TimingCalibrationStruct_h

#include "DQMServices/Core/interface/DQMStore.h"
#include <unordered_map>

struct TimingCalibrationHistograms {
public:
  TimingCalibrationHistograms() = default;

  using MonitorMap = std::unordered_map<uint32_t, dqm::reco::MonitorElement*>;

  MonitorMap leadingTime, toT;
  MonitorMap leadingTimeVsToT;
};

#endif
