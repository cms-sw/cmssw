#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh

/** \class EcalUncalibRecHitRecGlobalAlgo                                                                                                                                           
 *  Template used to compute amplitude, pedestal using a weights method                                                                                                            
 *                           time using a ratio method                                                                                                                             
 *                           chi2 using express method  
 *
 *  \author R. Bruneliere - A. Zabi
 */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecChi2Algo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRatioMethodAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

class EcalUncalibRecHitWorkerGlobal : public EcalUncalibRecHitWorkerRunOneDigiBase {
public:
  EcalUncalibRecHitWorkerGlobal(const edm::ParameterSet&, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerGlobal() : testbeamEEShape(EEShape(true)), testbeamEBShape(EBShape(true)) { ; }
  ~EcalUncalibRecHitWorkerGlobal() override{};

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt,
           const EcalDigiCollection::const_iterator& digi,
           EcalUncalibratedRecHitCollection& result) override;

  edm::ParameterSetDescription getAlgoDescription() override;

protected:
  double pedVec[3];
  double pedRMSVec[3];
  double gainRatios[3];

  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> tokenPeds_;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> tokenGains_;
  edm::ESHandle<EcalPedestals> peds_;
  edm::ESHandle<EcalGainRatios> gains_;

  template <class C>
  int isSaturated(const C& digi);

  double timeCorrection(float ampli, const std::vector<float>& amplitudeBins, const std::vector<float>& shiftBins);

  // weights method
  edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> tokenGrps_;
  edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> tokenWgts_;
  edm::ESHandle<EcalWeightXtalGroups> grps_;
  edm::ESHandle<EcalTBWeights> wgts_;
  const EcalWeightSet::EcalWeightMatrix* weights[2];
  const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];
  EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> weightsMethod_barrel_;
  EcalUncalibRecHitRecWeightsAlgo<EEDataFrame> weightsMethod_endcap_;
  EEShape testbeamEEShape;  // used in the chi2
  EBShape testbeamEBShape;  // can be replaced by simple shape arrays of float in the future

  // determie which of the samples must actually be used by ECAL local reco
  edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> tokenSampleMask_;
  edm::ESHandle<EcalSampleMask> sampleMaskHand_;

  // ratio method
  std::vector<double> EBtimeFitParameters_;
  std::vector<double> EEtimeFitParameters_;
  std::vector<double> EBamplitudeFitParameters_;
  std::vector<double> EEamplitudeFitParameters_;
  std::pair<double, double> EBtimeFitLimits_;
  std::pair<double, double> EEtimeFitLimits_;

  EcalUncalibRecHitRatioMethodAlgo<EBDataFrame> ratioMethod_barrel_;
  EcalUncalibRecHitRatioMethodAlgo<EEDataFrame> ratioMethod_endcap_;

  double EBtimeConstantTerm_;
  double EBtimeNconst_;
  double EEtimeConstantTerm_;
  double EEtimeNconst_;
  double outOfTimeThreshG12pEB_;
  double outOfTimeThreshG12mEB_;
  double outOfTimeThreshG61pEB_;
  double outOfTimeThreshG61mEB_;
  double outOfTimeThreshG12pEE_;
  double outOfTimeThreshG12mEE_;
  double outOfTimeThreshG61pEE_;
  double outOfTimeThreshG61mEE_;
  double amplitudeThreshEB_;
  double amplitudeThreshEE_;
  double ebSpikeThresh_;

  edm::ESGetToken<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd> tokenTimeCorrBias_;
  edm::ESHandle<EcalTimeBiasCorrections> timeCorrBias_;

  edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> tokenItime_;
  edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> tokenOfftime_;
  edm::ESHandle<EcalTimeCalibConstants> itime_;
  edm::ESHandle<EcalTimeOffsetConstant> offtime_;
  std::vector<double> ebPulseShape_;
  std::vector<double> eePulseShape_;

  // chi2 method
  bool kPoorRecoFlagEB_;
  bool kPoorRecoFlagEE_;
  double chi2ThreshEB_;
  double chi2ThreshEE_;
  std::vector<double> EBchi2Parameters_;
  std::vector<double> EEchi2Parameters_;
};

#endif
