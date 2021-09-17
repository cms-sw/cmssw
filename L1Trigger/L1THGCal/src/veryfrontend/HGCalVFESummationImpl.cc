#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"

HGCalVFESummationImpl::HGCalVFESummationImpl(const edm::ParameterSet& conf)
    : lsb_silicon_fC_(conf.getParameter<double>("siliconCellLSB_fC")),
      lsb_scintillator_MIP_(conf.getParameter<double>("scintillatorCellLSB_MIP")) {
  constexpr unsigned nThickness = 3;
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

void HGCalVFESummationImpl::triggerCellSums(const std::vector<std::pair<DetId, uint32_t>>& input_dataframes,
                                            std::unordered_map<uint32_t, uint32_t>& triggercells) {
  if (input_dataframes.empty())
    return;
  // sum energies in trigger cells
  for (const auto& [cellid, value] : input_dataframes) {
    // Apply noise threshold before summing into trigger cells
    uint32_t value_zero_suppressed = value;
    if (triggerTools_.isSilicon(cellid)) {
      int thickness = triggerTools_.thicknessIndex(cellid);
      double threshold = thresholds_silicon_.at(thickness);
      value_zero_suppressed = (value * lsb_silicon_fC_ > threshold ? value : 0);
    } else if (triggerTools_.isScintillator(cellid)) {
      value_zero_suppressed = (value * lsb_scintillator_MIP_ > threshold_scintillator_ ? value : 0);
    }
    if (value_zero_suppressed == 0)
      continue;

    // find trigger cell associated to cell
    uint32_t tcid = triggerTools_.getTriggerGeometry()->getTriggerCellFromCell(cellid);
    triggercells.emplace(tcid, 0);  // do nothing if key exists already

    // sums energy for the same trigger cell id
    triggercells[tcid] += value_zero_suppressed;  // 32 bits integer should be largely enough
  }
}
