#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerRatio_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerRatio_hh

/** \class EcalUncalibRecHitRecRatioAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  \author A. Ledovskoy (Design) - M. Balazs (Implementation)
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRatioMethodAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

class EcalUncalibRecHitWorkerRatio : public EcalUncalibRecHitWorkerRunOneDigiBase {
public:
  EcalUncalibRecHitWorkerRatio(const edm::ParameterSet&, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerRatio(){};
  ~EcalUncalibRecHitWorkerRatio() override{};

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt,
           const EcalDigiCollection::const_iterator& digi,
           EcalUncalibratedRecHitCollection& result) override;

  edm::ParameterSetDescription getAlgoDescription() override;

protected:
  // determie which of the samples must actually be used by ECAL local reco
  edm::ESHandle<EcalSampleMask> sampleMaskHand_;
  edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> sampleMaskToken_;

  edm::ESHandle<EcalPedestals> peds;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedsToken_;
  edm::ESHandle<EcalGainRatios> gains;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainsToken_;

  double pedVec[3];
  double pedRMSVec[3];
  double gainRatios[3];

  std::vector<double> EBtimeFitParameters_;
  std::vector<double> EEtimeFitParameters_;

  std::vector<double> EBamplitudeFitParameters_;
  std::vector<double> EEamplitudeFitParameters_;

  std::pair<double, double> EBtimeFitLimits_;
  std::pair<double, double> EEtimeFitLimits_;

  double EBtimeConstantTerm_;
  double EEtimeConstantTerm_;

  EcalUncalibRecHitRatioMethodAlgo<EBDataFrame> uncalibMaker_barrel_;
  EcalUncalibRecHitRatioMethodAlgo<EEDataFrame> uncalibMaker_endcap_;
};

#endif
