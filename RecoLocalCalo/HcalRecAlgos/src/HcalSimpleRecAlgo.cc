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
      ampl+=tool[i]*calibs.gain(capid)-calibs.pedestal(capid);
    }

    return RecHit(digi.id(),ampl,0);    
  }
}

HBHERecHit HcalSimpleRecAlgo::reconstruct(const HBHEDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HBHEDataFrame,HBHERecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
HFRecHit HcalSimpleRecAlgo::reconstruct(const HFDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HFDataFrame,HFRecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
HORecHit HcalSimpleRecAlgo::reconstruct(const HODataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HODataFrame,HORecHit>(digi,coder,calibs,firstSample_,samplesToAdd_);
}
