#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorCoarsenerImpl.h"

HGCalConcentratorCoarsenerImpl::HGCalConcentratorCoarsenerImpl(const edm::ParameterSet& conf)
    : fixedDataSizePerHGCROC_(conf.getParameter<bool>("fixedDataSizePerHGCROC")),
      coarseTCmapping_(conf.getParameter<std::vector<unsigned>>("ctcSize")),
      calibration_(conf.getParameterSet("superTCCalibration")),
      vfeCompression_(conf.getParameterSet("coarseTCCompression")) {}

void HGCalConcentratorCoarsenerImpl::updateCoarseTriggerCellMaps(const l1t::HGCalTriggerCell& tc, uint32_t ctcid) {
  auto& ctc = coarseTCs_[ctcid];

  ctc.sumPt += tc.pt();
  ctc.sumHwPt += tc.hwPt();
  ctc.sumMipPt += tc.mipPt();

  if (tc.mipPt() > ctc.maxMipPt) {
    ctc.maxId = tc.detId();
    ctc.maxMipPt = tc.mipPt();
  }
}

void HGCalConcentratorCoarsenerImpl::assignCoarseTriggerCellEnergy(l1t::HGCalTriggerCell& tc,
                                                                   const CoarseTC& ctc) const {
  //Compress and recalibrate CTC energy
  uint32_t code(0);
  uint64_t compressed_value(0);
  vfeCompression_.compressSingle(ctc.sumHwPt, code, compressed_value);

  if (compressed_value > std::numeric_limits<int>::max())
    edm::LogWarning("CompressedValueDowncasting") << "Compressed value cannot fit into 32-bit word. Downcasting.";

  tc.setHwPt(static_cast<int>(compressed_value));
  calibration_.calibrateInGeV(tc);
}

void HGCalConcentratorCoarsenerImpl::coarsen(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                             std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  coarseTCs_.clear();

  // first pass, fill the coarse trigger cell information
  for (const l1t::HGCalTriggerCell& tc : trigCellVecInput) {
    int thickness = triggerTools_.thicknessIndex(tc.detId());

    if (fixedDataSizePerHGCROC_ && thickness == kHighDensityThickness_) {
      trigCellVecOutput.push_back(tc);
      continue;
    }

    uint32_t ctcid = coarseTCmapping_.getCoarseTriggerCellId(tc.detId());
    updateCoarseTriggerCellMaps(tc, ctcid);
  }

  for (const auto& ctc : coarseTCs_) {
    l1t::HGCalTriggerCell triggerCell;

    uint32_t representativeId = coarseTCmapping_.getRepresentativeDetId(ctc.second.maxId);
    triggerCell.setDetId(representativeId);

    GlobalPoint point = coarseTCmapping_.getCoarseTriggerCellPosition(ctc.first);
    math::PtEtaPhiMLorentzVector initial_p4(ctc.second.sumPt, point.eta(), point.phi(), 0);

    triggerCell.setP4(initial_p4);
    assignCoarseTriggerCellEnergy(triggerCell, ctc.second);

    math::PtEtaPhiMLorentzVector p4(triggerCell.pt(), point.eta(), point.phi(), 0);
    triggerCell.setPosition(point);
    triggerCell.setP4(p4);
    trigCellVecOutput.push_back(triggerCell);
  }
}
