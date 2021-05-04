#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorThresholdImpl.h"

HGCalConcentratorThresholdImpl::HGCalConcentratorThresholdImpl(const edm::ParameterSet& conf)
    : threshold_silicon_(conf.getParameter<double>("threshold_silicon")),
      threshold_scintillator_(conf.getParameter<double>("threshold_scintillator")) {}

void HGCalConcentratorThresholdImpl::select(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                            std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput,
                                            std::vector<l1t::HGCalTriggerCell>& trigCellVecNotSelected) {
  for (const auto& trigCell : trigCellVecInput) {
    bool isScintillator = triggerTools_.isScintillator(trigCell.detId());
    double threshold = (isScintillator ? threshold_scintillator_ : threshold_silicon_);
    if (trigCell.mipPt() >= threshold) {
      trigCellVecOutput.push_back(trigCell);
    } else {
      trigCellVecNotSelected.push_back(trigCell);
    }
  }
}
