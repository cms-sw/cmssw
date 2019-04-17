#include "L1Trigger/L1THGCal/interface/veryfrontend/HGCalVFESummationImpl.h"

HGCalVFESummationImpl::HGCalVFESummationImpl(const edm::ParameterSet& conf)
    : thickness_corrections_(conf.getParameter<std::vector<double>>("ThicknessCorrections")) {}

void HGCalVFESummationImpl::triggerCellSums(const HGCalTriggerGeometryBase& geometry,
                                            const std::vector<std::pair<DetId, uint32_t>>& linearized_dataframes,
                                            std::unordered_map<uint32_t, uint32_t>& payload) {
  if (linearized_dataframes.empty())
    return;
  // sum energies in trigger cells
  for (const auto& frame : linearized_dataframes) {
    DetId cellid(frame.first);

    // find trigger cell associated to cell
    uint32_t tcid = geometry.getTriggerCellFromCell(cellid);
    payload.emplace(tcid, 0);  // do nothing if key exists already
    uint32_t value = frame.second;
    unsigned det = cellid.det();
    // equalize value among cell thicknesses for Silicon parts
    if (det == DetId::Forward || det == DetId::HGCalEE || det == DetId::HGCalHSi) {
      int thickness = triggerTools_.thicknessIndex(cellid);
      double thickness_correction = thickness_corrections_.at(thickness);
      value = (double)value * thickness_correction;
    }

    // sums energy for the same trigger cell id
    payload[tcid] += value;  // 32 bits integer should be largely enough
  }
}
