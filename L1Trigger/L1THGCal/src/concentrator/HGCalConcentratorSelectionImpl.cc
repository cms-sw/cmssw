#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorSelectionImpl.h"

HGCalConcentratorSelectionImpl::HGCalConcentratorSelectionImpl(const edm::ParameterSet& conf)
    : nData_(conf.getParameter<uint32_t>("NData")),
      nCellsInModule_(conf.getParameter<uint32_t>("MaxCellsInModule")),
      linLSB_(conf.getParameter<double>("linLSB")),
      adcsaturationBH_(conf.getParameter<double>("adcsaturationBH")),
      adcnBitsBH_(conf.getParameter<uint32_t>("adcnBitsBH")),
      TCThreshold_fC_(conf.getParameter<double>("TCThreshold_fC")),
      TCThresholdBH_MIP_(conf.getParameter<double>("TCThresholdBH_MIP")),
      triggercell_threshold_silicon_(conf.getParameter<double>("triggercell_threshold_silicon")),
      triggercell_threshold_scintillator_(conf.getParameter<double>("triggercell_threshold_scintillator")) {
  // Cannot have more selected cells than the max number of cells
  if (nData_ > nCellsInModule_)
    nData_ = nCellsInModule_;
  adcLSBBH_ = adcsaturationBH_ / pow(2., adcnBitsBH_);
  TCThreshold_ADC_ = (int)(TCThreshold_fC_ / linLSB_);
  TCThresholdBH_ADC_ = (int)(TCThresholdBH_MIP_ / adcLSBBH_);
}

void HGCalConcentratorSelectionImpl::thresholdSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                                         std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  for (const auto& trigCell : trigCellVecInput) {
    bool isScintillator = triggerTools_.isScintillator(trigCell.detId());
    int threshold = (isScintillator ? TCThresholdBH_ADC_ : TCThreshold_ADC_);
    double triggercell_threshold =
        (isScintillator ? triggercell_threshold_scintillator_ : triggercell_threshold_silicon_);

    if ((trigCell.hwPt() >= threshold) && (trigCell.mipPt() >= triggercell_threshold)) {
      trigCellVecOutput.push_back(trigCell);
    }
  }
}

void HGCalConcentratorSelectionImpl::bestChoiceSelectImpl(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                                          std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  trigCellVecOutput = trigCellVecInput;
  // sort, reverse order
  std::sort(trigCellVecOutput.begin(),
            trigCellVecOutput.end(),
            [](const l1t::HGCalTriggerCell& a, const l1t::HGCalTriggerCell& b) -> bool { return a.hwPt() > b.hwPt(); });

  // keep only the first trigger cells
  if (trigCellVecOutput.size() > nData_)
    trigCellVecOutput.resize(nData_);
}
