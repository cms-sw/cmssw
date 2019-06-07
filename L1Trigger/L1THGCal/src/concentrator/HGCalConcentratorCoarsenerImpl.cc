#include "L1Trigger/L1THGCal/interface/concentrator/HGCalConcentratorCoarsenerImpl.h"

HGCalConcentratorCoarsenerImpl::HGCalConcentratorCoarsenerImpl(const edm::ParameterSet& conf)
    : fixedDataSizePerHGCROC_(conf.getParameter<bool>("fixedDataSizePerHGCROC")),
      coarseTCmapping_(std::vector<unsigned>{HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_,
                                             HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_,
                                             HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_,
                                             HGCalCoarseTriggerCellMapping::kCTCsizeVeryFine_}) {}

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
  tc.setHwPt(ctc.sumHwPt);
  tc.setMipPt(ctc.sumMipPt);
  tc.setPt(ctc.sumPt);
}

void HGCalConcentratorCoarsenerImpl::coarsen(const std::vector<l1t::HGCalTriggerCell>& trigCellVecInput,
                                             std::vector<l1t::HGCalTriggerCell>& trigCellVecOutput) {
  coarseTCs_.clear();

  // first pass, fill the coarse trigger cell information
  for (const l1t::HGCalTriggerCell& tc : trigCellVecInput) {
    int thickness = 0;
    if (triggerTools_.isSilicon(tc.detId())) {
      thickness = triggerTools_.thicknessIndex(tc.detId(), true);
    } else if (triggerTools_.isScintillator(tc.detId())) {
      thickness = 3;
    }

    if (fixedDataSizePerHGCROC_ && thickness == kHighDensityThickness_) {
      trigCellVecOutput.push_back(tc);
      continue;
    }

    uint32_t ctcid = coarseTCmapping_.getCoarseTriggerCellId(tc.detId());
    updateCoarseTriggerCellMaps(tc, ctcid);
  }

  for (const auto& ctc : coarseTCs_) {
    l1t::HGCalTriggerCell triggerCell;
    assignCoarseTriggerCellEnergy(triggerCell, ctc.second);
    uint32_t representativeId = coarseTCmapping_.getRepresentativeDetId(ctc.second.maxId);
    triggerCell.setDetId(representativeId);
    GlobalPoint point = coarseTCmapping_.getCoarseTriggerCellPosition(ctc.first);
    math::PtEtaPhiMLorentzVector p4(triggerCell.pt(), point.eta(), point.phi(), 0);
    triggerCell.setPosition(point);
    triggerCell.setP4(p4);
    trigCellVecOutput.push_back(triggerCell);
  }
}
