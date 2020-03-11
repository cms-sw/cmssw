#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"

HGCalVFESummationImpl::HGCalVFESummationImpl(const edm::ParameterSet& conf)
    : thickness_corrections_(conf.getParameter<std::vector<double>>("ThicknessCorrections")),
      lsb_silicon_fC_(conf.getParameter<double>("siliconCellLSB_fC")),
      lsb_scintillator_MIP_(conf.getParameter<double>("scintillatorCellLSB_MIP")) {
  const unsigned nThickness = 3;
  if (thickness_corrections_.size() != nThickness) {
    throw cms::Exception("Configuration")
        << thickness_corrections_.size() << " thickness corrections are given instead of " << nThickness
        << " (the number of sensor thicknesses)";
  }
  thresholds_silicon_ =
      conf.getParameter<edm::ParameterSet>("noiseSilicon").getParameter<std::vector<double>>("values");
  if (thresholds_silicon_.size() != nThickness) {
    throw cms::Exception("Configuration") << thresholds_silicon_.size() << " silicon thresholds are given instead of "
                                          << nThickness << " (the number of sensor thicknesses)";
  }
  threshold_scintillator_ = conf.getParameter<edm::ParameterSet>("noiseScintillator").getParameter<double>("noise_MIP");
  const auto threshold = conf.getParameter<double>("noiseThreshold");
  std::transform(
      thresholds_silicon_.begin(), thresholds_silicon_.end(), thresholds_silicon_.begin(), [threshold](auto noise) {
        return noise * threshold;
      });
  threshold_scintillator_ *= threshold;
}

void HGCalVFESummationImpl::triggerCellSums(const HGCalTriggerGeometryBase& geometry,
                                            const std::vector<std::pair<DetId, uint32_t>>& linearized_dataframes,
                                            std::unordered_map<uint32_t, uint32_t>& payload) {
  if (linearized_dataframes.empty())
    return;
  // sum energies in trigger cells
  for (const auto& frame : linearized_dataframes) {
    DetId cellid(frame.first);
    uint32_t value = frame.second;

    // Apply noise threshold before summing into trigger cells
    if (triggerTools_.isSilicon(cellid)) {
      int thickness = triggerTools_.thicknessIndex(cellid);
      double threshold = thresholds_silicon_.at(thickness);
      value = (value * lsb_silicon_fC_ > threshold ? value : 0);
    } else if (triggerTools_.isScintillator(cellid)) {
      value = (value * lsb_scintillator_MIP_ > threshold_scintillator_ ? value : 0);
    }
    if (value == 0)
      continue;

    // find trigger cell associated to cell
    uint32_t tcid = geometry.getTriggerCellFromCell(cellid);
    payload.emplace(tcid, 0);  // do nothing if key exists already

    // equalize value among cell thicknesses for Silicon parts
    if (triggerTools_.isSilicon(cellid) && !triggerTools_.isNose(cellid)) {
      int thickness = triggerTools_.thicknessIndex(cellid);
      double thickness_correction = thickness_corrections_.at(thickness);
      value = (double)value * thickness_correction;
    }

    // sums energy for the same trigger cell id
    payload[tcid] += value;  // 32 bits integer should be largely enough
  }
}
