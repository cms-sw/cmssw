#ifndef HCALHITRECONSTRUCTOR_H
#define HCALHITRECONSTRUCTOR_H 1

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromRecHits.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHFStatusBitFromDigis.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParam.h"
#include "CondFormats/HcalObjects/interface/HcalFlagHFDigiTimeParams.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimingCorrector.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalADCSaturationFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFTimingTrustFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_S9S1algorithm.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_PETalgorithm.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

/** \class HcalHitReconstructor
	
    \author J. Temple & E. Yazgan
    ** Based on HcalSimpleReconstructor.h by J. Mans
    */

class HcalTopology;

class HcalHitReconstructor : public edm::stream::EDProducer<> {
public:
  explicit HcalHitReconstructor(const edm::ParameterSet& ps);
  ~HcalHitReconstructor() override;

  void beginRun(edm::Run const& r, edm::EventSetup const& es) final;
  void endRun(edm::Run const& r, edm::EventSetup const& es) final;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  typedef void (HcalSimpleRecAlgo::*SetCorrectionFcn)(std::shared_ptr<AbsOOTPileupCorrection>);

  HcalSimpleRecAlgo reco_;
  HcalADCSaturationFlag* saturationFlagSetter_;
  HFTimingTrustFlag* HFTimingTrustFlagSetter_;
  HcalHFStatusBitFromDigis* hfdigibit_;
  HcalHF_S9S1algorithm* hfS9S1_;
  HcalHF_S9S1algorithm* hfS8S1_;
  HcalHF_PETalgorithm* hfPET_;

  DetId::Detector det_;
  int subdet_;
  HcalOtherSubdetector subdetOther_;
  edm::InputTag inputLabel_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;
  edm::EDGetTokenT<HFDigiCollection> tok_hf_;
  edm::EDGetTokenT<HcalCalibDigiCollection> tok_calib_;
  //std::vector<std::string> channelStatusToDrop_;
  bool correctTiming_;        // turn on/off Ken Rossato's algorithm to fix timing
  bool setNoiseFlags_;        // turn on/off basic noise flags
  bool setHSCPFlags_;         // turn on/off HSCP noise flags
  bool setSaturationFlags_;   // turn on/off flag indicating ADC saturation
  bool setTimingTrustFlags_;  // turn on/off HF timing uncertainty flag
  bool setPulseShapeFlags_;   //  turn on/off HBHE fit-based noise flags
  bool setNegativeFlags_;     // turn on/off HBHE negative noise flags
  bool dropZSmarkedPassed_;   // turn on/off dropping of zero suppression marked and passed digis

  int firstAuxTS_;

  // legacy parameters for config-set values compatibility
  int firstSample_;
  int samplesToAdd_;
  bool tsFromDB_;
  bool recoParamsFromDB_;
  bool digiTimeFromDB_;

  // switch on/off leakage (to pre-sample) correction
  bool useLeakCorrection_;

  // Labels related to OOT pileup corrections
  std::string dataOOTCorrectionName_;
  std::string dataOOTCorrectionCategory_;
  std::string mcOOTCorrectionName_;
  std::string mcOOTCorrectionCategory_;
  SetCorrectionFcn setPileupCorrection_;

  HcalRecoParams* paramTS;                                     // firstSample & sampleToAdd from DB
  std::unique_ptr<HcalFlagHFDigiTimeParams> HFDigiTimeParams;  // HF DigiTime parameters

  std::string corrName_, cat_;
};

#endif
