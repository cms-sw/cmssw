#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh

/** \class EcalUncalibRecHitRecGlobalAlgo                                                                                                                                           
 *  Template used to compute amplitude, pedestal using a weights method                                                                                                            
 *                           time using a ratio method                                                                                                                             
 *                           chi2 using express method  
 *
 *  \author R. Bruneliere - A. Zabi
 */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimeWeightsAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecChi2Algo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRatioMethodAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimingCCAlgo.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

class EcalUncalibRecHitWorkerMultiFit final : public EcalUncalibRecHitWorkerBaseClass {
public:
  EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerMultiFit(){};
  ~EcalUncalibRecHitWorkerMultiFit() override{};

private:
  void set(const edm::EventSetup& es) override;
  void set(const edm::Event& evt) override;
  void run(const edm::Event& evt, const EcalDigiCollection& digis, EcalUncalibratedRecHitCollection& result) override;

public:
  edm::ParameterSetDescription getAlgoDescription() override;

private:
  edm::ESHandle<EcalPedestals> peds;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedsToken_;
  edm::ESHandle<EcalGainRatios> gains;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainsToken_;
  edm::ESHandle<EcalSamplesCorrelation> noisecovariances;
  edm::ESGetToken<EcalSamplesCorrelation, EcalSamplesCorrelationRcd> noiseConvariancesToken_;
  edm::ESHandle<EcalPulseShapes> pulseshapes;
  edm::ESGetToken<EcalPulseShapes, EcalPulseShapesRcd> pulseShapesToken_;
  edm::ESHandle<EcalPulseCovariances> pulsecovariances;
  edm::ESGetToken<EcalPulseCovariances, EcalPulseCovariancesRcd> pulseConvariancesToken_;

  double timeCorrection(float ampli, const std::vector<float>& amplitudeBins, const std::vector<float>& shiftBins);

  const SampleMatrix& noisecor(bool barrel, int gain) const { return noisecors_[barrel ? 1 : 0][gain]; }
  const SampleMatrixGainArray& noisecor(bool barrel) const { return noisecors_[barrel ? 1 : 0]; }

  // multifit method
  std::array<SampleMatrixGainArray, 2> noisecors_;
  BXVector activeBX;
  bool ampErrorCalculation_;
  bool useLumiInfoRunHeader_;
  EcalUncalibRecHitMultiFitAlgo multiFitMethod_;

  int bunchSpacingManual_;
  edm::EDGetTokenT<unsigned int> bunchSpacing_;

  // determine which of the samples must actually be used by ECAL local reco
  edm::ESHandle<EcalSampleMask> sampleMaskHand_;
  edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> sampleMaskToken_;

  // time algorithm to be used to set the jitter and its uncertainty
  enum TimeAlgo { noMethod, ratioMethod, weightsMethod, crossCorrelationMethod };
  TimeAlgo timealgo_ = noMethod;

  // time weights method
  edm::ESHandle<EcalWeightXtalGroups> grps;
  edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> grpsToken_;
  edm::ESHandle<EcalTBWeights> wgts;
  edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> wgtsToken_;
  const EcalWeightSet::EcalWeightMatrix* weights[2];
  EcalUncalibRecHitTimeWeightsAlgo<EBDataFrame> weightsMethod_barrel_;
  EcalUncalibRecHitTimeWeightsAlgo<EEDataFrame> weightsMethod_endcap_;
  bool doPrefitEB_;
  bool doPrefitEE_;
  double prefitMaxChiSqEB_;
  double prefitMaxChiSqEE_;
  bool dynamicPedestalsEB_;
  bool dynamicPedestalsEE_;
  bool mitigateBadSamplesEB_;
  bool mitigateBadSamplesEE_;
  bool gainSwitchUseMaxSampleEB_;
  bool gainSwitchUseMaxSampleEE_;
  bool selectiveBadSampleCriteriaEB_;
  bool selectiveBadSampleCriteriaEE_;
  double addPedestalUncertaintyEB_;
  double addPedestalUncertaintyEE_;
  bool simplifiedNoiseModelForGainSwitch_;

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
  double EEtimeConstantTerm_;
  double EBtimeNconst_;
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

  edm::ESHandle<EcalTimeBiasCorrections> timeCorrBias_;
  edm::ESGetToken<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd> timeCorrBiasToken_;

  edm::ESHandle<EcalTimeCalibConstants> itime;
  edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> itimeToken_;
  edm::ESHandle<EcalTimeOffsetConstant> offtime;
  edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> offtimeToken_;
  std::vector<double> ebPulseShape_;
  std::vector<double> eePulseShape_;

  // chi2 thresholds for flags settings
  bool kPoorRecoFlagEB_;
  bool kPoorRecoFlagEE_;
  double chi2ThreshEB_;
  double chi2ThreshEE_;

  //Timing Cross Correlation Algo
  std::unique_ptr<EcalUncalibRecHitTimingCCAlgo> computeCC_;
};

#endif
