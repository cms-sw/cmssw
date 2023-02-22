//
// Created: 2 Mar 2006
//          Shahram Rahatlou, University of Rome & INFN
//
#ifndef CalibCalorimetry_EcalPlugins_EcalTrivialConditionRetriever_H
#define CalibCalorimetry_EcalPlugins_EcalTrivialConditionRetriever_H
// system include files
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSimComponentShape.h"
#include "CondFormats/DataRecord/interface/EcalSimComponentShapeRcd.h"

#include "CondFormats/EcalObjects/interface/EcalSimPulseShape.h"
#include "CondFormats/DataRecord/interface/EcalSimPulseShapeRcd.h"

#include "SimG4CMS/Calo/interface/EnergyResolutionVsLumi.h"
#include "SimG4CMS/Calo/interface/EvolutionECAL.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
// forward declarations

namespace edm {
  class ParameterSet;
}

class EcalTrivialConditionRetriever : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  EcalTrivialConditionRetriever(const edm::ParameterSet& pset);
  EcalTrivialConditionRetriever(const EcalTrivialConditionRetriever&) = delete;                   // stop default
  const EcalTrivialConditionRetriever& operator=(const EcalTrivialConditionRetriever&) = delete;  // stop default
  ~EcalTrivialConditionRetriever() override;

  // ---------- member functions ---------------------------
  virtual std::unique_ptr<EcalPedestals> produceEcalPedestals(const EcalPedestalsRcd&);
  virtual std::unique_ptr<EcalWeightXtalGroups> produceEcalWeightXtalGroups(const EcalWeightXtalGroupsRcd&);
  virtual std::unique_ptr<EcalLinearCorrections> produceEcalLinearCorrections(const EcalLinearCorrectionsRcd&);
  virtual std::unique_ptr<EcalIntercalibConstants> produceEcalIntercalibConstants(const EcalIntercalibConstantsRcd&);
  virtual std::unique_ptr<EcalIntercalibConstantsMC> produceEcalIntercalibConstantsMC(
      const EcalIntercalibConstantsMCRcd&);
  virtual std::unique_ptr<EcalIntercalibErrors> produceEcalIntercalibErrors(const EcalIntercalibErrorsRcd&);
  virtual std::unique_ptr<EcalTimeCalibConstants> produceEcalTimeCalibConstants(const EcalTimeCalibConstantsRcd&);
  virtual std::unique_ptr<EcalTimeCalibErrors> produceEcalTimeCalibErrors(const EcalTimeCalibErrorsRcd&);
  virtual std::unique_ptr<EcalGainRatios> produceEcalGainRatios(const EcalGainRatiosRcd&);
  virtual std::unique_ptr<EcalADCToGeVConstant> produceEcalADCToGeVConstant(const EcalADCToGeVConstantRcd&);
  virtual std::unique_ptr<EcalTBWeights> produceEcalTBWeights(const EcalTBWeightsRcd&);
  virtual std::unique_ptr<EcalIntercalibConstants> getIntercalibConstantsFromConfiguration(
      const EcalIntercalibConstantsRcd&);
  virtual std::unique_ptr<EcalIntercalibConstantsMC> getIntercalibConstantsMCFromConfiguration(
      const EcalIntercalibConstantsMCRcd&);
  virtual std::unique_ptr<EcalSimComponentShape> getEcalSimComponentShapeFromConfiguration(
      const EcalSimComponentShapeRcd&);
  virtual std::unique_ptr<EcalSimPulseShape> getEcalSimPulseShapeFromConfiguration(const EcalSimPulseShapeRcd&);
  virtual std::unique_ptr<EcalIntercalibErrors> getIntercalibErrorsFromConfiguration(const EcalIntercalibErrorsRcd&);
  virtual std::unique_ptr<EcalTimeCalibConstants> getTimeCalibConstantsFromConfiguration(
      const EcalTimeCalibConstantsRcd&);
  virtual std::unique_ptr<EcalTimeCalibErrors> getTimeCalibErrorsFromConfiguration(const EcalTimeCalibErrorsRcd&);
  virtual std::unique_ptr<EcalTimeOffsetConstant> produceEcalTimeOffsetConstant(const EcalTimeOffsetConstantRcd&);

  virtual std::unique_ptr<EcalLaserAlphas> produceEcalLaserAlphas(const EcalLaserAlphasRcd&);
  virtual std::unique_ptr<EcalLaserAPDPNRatiosRef> produceEcalLaserAPDPNRatiosRef(const EcalLaserAPDPNRatiosRefRcd&);
  virtual std::unique_ptr<EcalLaserAPDPNRatios> produceEcalLaserAPDPNRatios(const EcalLaserAPDPNRatiosRcd&);

  virtual std::unique_ptr<EcalClusterLocalContCorrParameters> produceEcalClusterLocalContCorrParameters(
      const EcalClusterLocalContCorrParametersRcd&);
  virtual std::unique_ptr<EcalClusterCrackCorrParameters> produceEcalClusterCrackCorrParameters(
      const EcalClusterCrackCorrParametersRcd&);
  virtual std::unique_ptr<EcalClusterEnergyCorrectionParameters> produceEcalClusterEnergyCorrectionParameters(
      const EcalClusterEnergyCorrectionParametersRcd&);
  virtual std::unique_ptr<EcalClusterEnergyUncertaintyParameters> produceEcalClusterEnergyUncertaintyParameters(
      const EcalClusterEnergyUncertaintyParametersRcd&);
  virtual std::unique_ptr<EcalClusterEnergyCorrectionObjectSpecificParameters>
  produceEcalClusterEnergyCorrectionObjectSpecificParameters(
      const EcalClusterEnergyCorrectionObjectSpecificParametersRcd&);
  virtual std::unique_ptr<EcalPFRecHitThresholds> produceEcalPFRecHitThresholds(const EcalPFRecHitThresholdsRcd&);
  virtual std::unique_ptr<EcalPFRecHitThresholds> getPFRecHitThresholdsFromConfiguration(
      const EcalPFRecHitThresholdsRcd&);

  virtual std::unique_ptr<EcalChannelStatus> produceEcalChannelStatus(const EcalChannelStatusRcd&);
  virtual std::unique_ptr<EcalChannelStatus> getChannelStatusFromConfiguration(const EcalChannelStatusRcd&);

  virtual std::unique_ptr<EcalTPGCrystalStatus> produceEcalTrgChannelStatus(const EcalTPGCrystalStatusRcd&);
  virtual std::unique_ptr<EcalTPGCrystalStatus> getTrgChannelStatusFromConfiguration(const EcalTPGCrystalStatusRcd&);

  virtual std::unique_ptr<EcalDCSTowerStatus> produceEcalDCSTowerStatus(const EcalDCSTowerStatusRcd&);
  virtual std::unique_ptr<EcalDAQTowerStatus> produceEcalDAQTowerStatus(const EcalDAQTowerStatusRcd&);
  virtual std::unique_ptr<EcalDQMTowerStatus> produceEcalDQMTowerStatus(const EcalDQMTowerStatusRcd&);
  virtual std::unique_ptr<EcalDQMChannelStatus> produceEcalDQMChannelStatus(const EcalDQMChannelStatusRcd&);

  virtual std::unique_ptr<EcalMappingElectronics> produceEcalMappingElectronics(const EcalMappingElectronicsRcd&);
  virtual std::unique_ptr<EcalMappingElectronics> getMappingFromConfiguration(const EcalMappingElectronicsRcd&);

  //  virtual std::unique_ptr<EcalAlignmentEB> produceEcalAlignmentEB( const EcalAlignmentEBRcd& );
  //  virtual std::unique_ptr<EcalAlignmentEE> produceEcalAlignmentEE( const EcalAlignmentEERcd& );
  //  virtual std::unique_ptr<EcalAlignmentES> produceEcalAlignmentES( const EcalAlignmentESRcd& );
  virtual std::unique_ptr<Alignments> produceEcalAlignmentEB(const EBAlignmentRcd&);
  virtual std::unique_ptr<Alignments> produceEcalAlignmentEE(const EEAlignmentRcd&);
  virtual std::unique_ptr<Alignments> produceEcalAlignmentES(const ESAlignmentRcd&);

  virtual std::unique_ptr<EcalSampleMask> produceEcalSampleMask(const EcalSampleMaskRcd&);

  virtual std::unique_ptr<EcalTimeBiasCorrections> produceEcalTimeBiasCorrections(const EcalTimeBiasCorrectionsRcd&);

  virtual std::unique_ptr<EcalSamplesCorrelation> produceEcalSamplesCorrelation(const EcalSamplesCorrelationRcd&);

protected:
  //overriding from ContextRecordIntervalFinder
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  void getWeightsFromConfiguration(const edm::ParameterSet& ps);

  // data members
  double adcToGeVEBConstant_;  // ADC -> GeV scale for barrel
  double adcToGeVEEConstant_;  // ADC -> GeV scale for endcap

  double intercalibConstantMean_;   // mean of intercalib constant. default: 1.0
  double intercalibConstantSigma_;  // sigma of intercalib constant

  double intercalibConstantMeanMC_;   // mean of intercalib constant. default: 1.0
  double intercalibConstantSigmaMC_;  // sigma of intercalib constant
                                      // Gaussian used to generate intercalib constants for
                                      // each channel. no smearing if sigma=0.0 (default)
  double intercalibErrorMean_;        // mean of intercalib constant error

  double timeCalibConstantMean_;
  double timeCalibConstantSigma_;
  double timeCalibErrorMean_;

  double timeOffsetEBConstant_;
  double timeOffsetEEConstant_;

  // cluster corrections
  std::vector<double> localContCorrParameters_;
  std::vector<double> crackCorrParameters_;
  std::vector<double> energyCorrectionParameters_;
  std::vector<double> energyUncertaintyParameters_;
  std::vector<double> energyCorrectionObjectSpecificParameters_;
  double pfRecHitThresholdsNSigmas_;
  double pfRecHitThresholdsNSigmasHEta_;
  double pfRecHitThresholdsEB_;
  double pfRecHitThresholdsEE_;

  double sim_pulse_shape_EB_thresh_;
  double sim_pulse_shape_EE_thresh_;
  double sim_pulse_shape_APD_thresh_;
  float sim_pulse_shape_TI_;

  double laserAlphaMeanEER_higheta_;
  double laserAlphaMeanEEC_higheta_;

  std::string pfRecHitFile_;
  std::string pfRecHitFileEE_;

  std::string EELaserAlphaFile2_;

  // ageing parameters
  double totLumi_;
  double instLumi_;

  // laser
  double laserAlphaMean_;
  double laserAlphaSigma_;
  double laserAlphaMeanEBR_;
  double laserAlphaMeanEBC_;
  double laserAlphaMeanEER_;
  double laserAlphaMeanEEC_;

  double laserAPDPNRefMean_;
  double laserAPDPNRefSigma_;
  double laserAPDPNMean_;
  double laserAPDPNSigma_;
  unsigned long laserAPDPNTime1_;
  unsigned long laserAPDPNTime2_;
  unsigned long laserAPDPNTime3_;

  double linCorrMean_;   // mean of lin corr
  double linCorrSigma_;  // sigma of lin corr

  unsigned long linearTime1_;
  unsigned long linearTime2_;
  unsigned long linearTime3_;

  double EBpedMeanX12_;  // pedestal mean pedestal at gain 12
  double EBpedRMSX12_;   // pedestal rms at gain 12
  double EBpedMeanX6_;   // pedestal mean pedestal at gain 6
  double EBpedRMSX6_;    // pedestal rms at gain 6
  double EBpedMeanX1_;   // pedestal mean pedestal at gain 1
  double EBpedRMSX1_;    // pedestal rms at gain 1

  double EEpedMeanX12_;  // pedestal mean pedestal at gain 12
  double EEpedRMSX12_;   // pedestal rms at gain 12
  double EEpedMeanX6_;   // pedestal mean pedestal at gain 6
  double EEpedRMSX6_;    // pedestal rms at gain 6
  double EEpedMeanX1_;   // pedestal mean pedestal at gain 1
  double EEpedRMSX1_;    // pedestal rms at gain 1

  double gainRatio12over6_;  // ratio of MGPA gain12 / gain6
  double gainRatio6over1_;   // ratio of MGPA gain6 / gain1

  std::vector<ROOT::Math::SVector<double, EcalDataFrame::MAXSAMPLES> >
      amplWeights_;  // weights to compute amplitudes after ped subtraction
  std::vector<ROOT::Math::SVector<double, EcalDataFrame::MAXSAMPLES> >
      amplWeightsAft_;  // weights to compute amplitudes after ped subtraction

  std::vector<ROOT::Math::SVector<double, EcalDataFrame::MAXSAMPLES> >
      pedWeights_;  // weights to compute amplitudes w/o ped subtraction
  std::vector<ROOT::Math::SVector<double, EcalDataFrame::MAXSAMPLES> >
      pedWeightsAft_;  // weights to compute amplitudes w/o ped subtraction

  std::vector<ROOT::Math::SVector<double, EcalDataFrame::MAXSAMPLES> > jittWeights_;     // weights to compute jitter
  std::vector<ROOT::Math::SVector<double, EcalDataFrame::MAXSAMPLES> > jittWeightsAft_;  // weights to compute jitter

  std::vector<EcalWeightSet::EcalChi2WeightMatrix> chi2Matrix_;
  std::vector<EcalWeightSet::EcalChi2WeightMatrix> chi2MatrixAft_;

  std::string amplWeightsFile_;
  std::string amplWeightsAftFile_;
  std::string pedWeightsFile_;
  std::string pedWeightsAftFile_;
  std::string jittWeightsFile_;
  std::string jittWeightsAftFile_;
  std::string chi2MatrixFile_;
  std::string chi2MatrixAftFile_;
  std::string linearCorrectionsFile_;
  std::string intercalibConstantsFile_;
  std::string intercalibConstantsMCFile_;
  std::string intercalibErrorsFile_;
  std::string timeCalibConstantsFile_;
  std::string timeCalibErrorsFile_;
  std::string channelStatusFile_;
  std::string trgChannelStatusFile_;
  std::string mappingFile_;
  std::string EBAlignmentFile_;
  std::string EEAlignmentFile_;
  std::string ESAlignmentFile_;
  std::string EBLaserAlphaFile_;
  std::string EELaserAlphaFile_;
  unsigned int sampleMaskEB_;  // Mask to discard sample in barrel
  unsigned int sampleMaskEE_;  // Mask to discard sample in endcaps
  std::vector<double> EBtimeCorrAmplitudeBins_;
  std::vector<double> EBtimeCorrShiftBins_;
  std::vector<double> EEtimeCorrAmplitudeBins_;
  std::vector<double> EEtimeCorrShiftBins_;

  std::vector<double> EBG12samplesCorrelation_;
  std::vector<double> EBG6samplesCorrelation_;
  std::vector<double> EBG1samplesCorrelation_;
  std::vector<double> EEG12samplesCorrelation_;
  std::vector<double> EEG6samplesCorrelation_;
  std::vector<double> EEG1samplesCorrelation_;
  std::string SamplesCorrelationFile_;
  std::string EBSimPulseShapeFile_;
  std::string EESimPulseShapeFile_;
  std::string APDSimPulseShapeFile_;
  std::vector<std::string> EBSimComponentShapeFiles_;

  int nTDCbins_;

  bool getWeightsFromFile_;
  bool weightsForAsynchronousRunning_;
  bool producedEcalPedestals_;
  bool producedEcalWeights_;
  bool producedEcalLinearCorrections_;
  bool producedEcalIntercalibConstants_;
  bool producedEcalIntercalibConstantsMC_;
  bool producedEcalIntercalibErrors_;
  bool producedEcalTimeCalibConstants_;
  bool producedEcalTimeCalibErrors_;
  bool producedEcalTimeOffsetConstant_;
  bool producedEcalGainRatios_;
  bool producedEcalADCToGeVConstant_;
  bool producedEcalLaserCorrection_;
  bool producedEcalChannelStatus_;
  bool producedEcalDQMTowerStatus_;
  bool producedEcalDQMChannelStatus_;
  bool producedEcalDCSTowerStatus_;
  bool producedEcalDAQTowerStatus_;
  bool producedEcalTrgChannelStatus_;
  bool producedEcalClusterLocalContCorrParameters_;
  bool producedEcalClusterCrackCorrParameters_;
  bool producedEcalClusterEnergyCorrectionParameters_;
  bool producedEcalClusterEnergyUncertaintyParameters_;
  bool producedEcalClusterEnergyCorrectionObjectSpecificParameters_;
  bool producedEcalMappingElectronics_;
  bool producedEcalAlignmentEB_;
  bool producedEcalAlignmentEE_;
  bool producedEcalAlignmentES_;
  bool producedEcalSimComponentShape_;
  bool producedEcalSimPulseShape_;
  bool producedEcalPFRecHitThresholds_;
  bool getEBAlignmentFromFile_;
  bool getEEAlignmentFromFile_;
  bool getESAlignmentFromFile_;

  bool getSimComponentShapeFromFile_;
  bool getSimPulseShapeFromFile_;

  bool getLaserAlphaFromFileEB_;
  bool getLaserAlphaFromFileEE_;
  bool getLaserAlphaFromTypeEB_;
  bool getLaserAlphaFromTypeEE_;
  bool producedEcalSampleMask_;
  bool producedEcalTimeBiasCorrections_;
  bool producedEcalSamplesCorrelation_;
  bool getSamplesCorrelationFromFile_;

  int verbose_;  // verbosity
};
#endif
