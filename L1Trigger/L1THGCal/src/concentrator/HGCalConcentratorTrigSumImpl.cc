#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorTrigSumImpl.h"

HGCalConcentratorTrigSumImpl::HGCalConcentratorTrigSumImpl(const edm::ParameterSet& conf) {}

void HGCalConcentratorTrigSumImpl::doSum(uint32_t module_id,
                                         const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                         std::vector<l1t::HGCalTriggerSums>& trigSumsVecOutput) const {
  double ptsum = 0;
  double mipptsum = 0;
  double hwptsum = 0;

  for (const auto& trigCell : trigCellVecInput) {
    // detId selection is already done in HGCalConcentratorProcessorSelection:
    // here we do not worry about it and assume all cells are from the same module
    ptsum += trigCell.pt();
    mipptsum += trigCell.mipPt();
    hwptsum += trigCell.hwPt();
  }
  if (!trigCellVecInput.empty()) {
    GlobalPoint module_pos = triggerTools_.getTriggerGeometry()->getModulePosition(module_id);

    math::PtEtaPhiMLorentzVector p4(ptsum, module_pos.eta(), module_pos.phi(), 0);
    l1t::HGCalTriggerSums ts;
    ts.setP4(p4);
    ts.setDetId(module_id);
    ts.setPosition(module_pos);
    ts.setMipPt(mipptsum);
    ts.setHwPt(hwptsum);
    trigSumsVecOutput.push_back(ts);
  }
}
