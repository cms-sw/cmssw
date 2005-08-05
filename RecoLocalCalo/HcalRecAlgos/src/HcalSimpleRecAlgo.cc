#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"

HcalSimpleRecAlgo::HcalSimpleRecAlgo(int firstSample, int samplesToAdd) : firstSample_(firstSample), samplesToAdd_(samplesToAdd) {
}

namespace HcalSimpleRecAlgoImpl {
  template<class Digi, class RecHit>
  inline RecHit reco(const Digi& digi, const HcalCoder& coder, const HcalCalibrations& calibs, int ifirst, int n) {
    CaloSamples tool;
    coder.adc2fC(digi,tool);

    double ampl=0;
    for (int i=ifirst; i<tool.size() && i<n+ifirst; i++) {
      int capid=digi[i].capid();
      ampl+=(tool[i]-calibs.pedestal(capid))*calibs.gain(capid);
    }

    return RecHit(digi.id(),ampl,0);    
  }
}

cms::HBHERecHit HcalSimpleRecAlgo::reconstruct(const cms::HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<cms::HBHEDataFrame,cms::HBHERecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
cms::HFRecHit HcalSimpleRecAlgo::reconstruct(const cms::HFDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<cms::HFDataFrame,cms::HFRecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
cms::HORecHit HcalSimpleRecAlgo::reconstruct(const cms::HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<cms::HODataFrame,cms::HORecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
