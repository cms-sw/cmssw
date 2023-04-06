#ifndef ECALDBCOPY_H
#define ECALDBCOPY_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class EcalADCToGeVConstant;
class EcalTPGTowerStatus;
class EcalTBWeights;
class EcalLaserAPDPNRatios;
class Alignments;
class EcalTimeOffsetConstant;
class EcalSampleMask;
class EcalSimComponentShape;
class EcalSimPulseShape;
class EcalTimeBiasCorrections;
class EcalSamplesCorrelation;

class EcalPedestalsRcd;
class EcalADCToGeVConstantRcd;
class EcalTimeCalibConstantsRcd;
class EcalChannelStatusRcd;
class EcalDQMChannelStatusRcd;
class EcalDQMTowerStatusRcd;
class EcalDCSTowerStatusRcd;
class EcalTPGCrystalStatusRcd;
class EcalDAQTowerStatusRcd;
class EcalTPGTowerStatusRcd;
class EcalTPGTowerStatusRcd;
class EcalIntercalibConstantsRcd;
class EcalLinearCorrectionsRcd;
class EcalIntercalibConstantsMCRcd;
class EcalIntercalibErrorsRcd;
class EcalGainRatiosRcd;
class EcalWeightXtalGroupsRcd;
class EcalTBWeightsRcd;
class EcalLaserAlphasRcd;
class EcalLaserAPDPNRatiosRcd;
class EcalLaserAPDPNRatiosRefRcd;
class EcalClusterCrackCorrParametersRcd;
class EcalPFRecHitThresholdsRcd;
class EcalClusterEnergyUncertaintyParametersRcd;
class EcalClusterEnergyCorrectionParametersRcd;
class EcalClusterEnergyCorrectionObjectSpecificParametersRcd;
class EcalClusterLocalContCorrParametersRcd;
class EBAlignmentRcd;
class EEAlignmentRcd;
class ESAlignmentRcd;
class EcalTimeOffsetConstantRcd;
class EcalSampleMaskRcd;
class EcalSimComponentShapeRcd;
class EcalSimPulseShapeRcd;
class EcalTimeBiasCorrectionsRcd;
class EcalSamplesCorrelationRcd;

class EcalDBCopy : public edm::one::EDAnalyzer<> {
public:
  explicit EcalDBCopy(const edm::ParameterSet& iConfig);
  ~EcalDBCopy() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;

private:
  bool shouldCopy(const edm::EventSetup& evtSetup, const std::string& container);
  void copyToDB(const edm::EventSetup& evtSetup, const std::string& container);

  std::string m_timetype;
  std::map<std::string, unsigned long long> m_cacheIDs;
  std::map<std::string, std::string> m_records;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> ecalPedestalToken_;
  edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> ecalADCtoGeVToken_;
  edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> ecalTimeCalibToken_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> ecalChannelStatusToken_;
  edm::ESGetToken<EcalDQMChannelStatus, EcalDQMChannelStatusRcd> ecalDQMChannelStatusToken_;
  edm::ESGetToken<EcalDQMTowerStatus, EcalDQMTowerStatusRcd> ecalDQMTowerStatusToken_;
  edm::ESGetToken<EcalDCSTowerStatus, EcalDCSTowerStatusRcd> ecalDCSTowerStatusToken_;
  edm::ESGetToken<EcalDAQTowerStatus, EcalDAQTowerStatusRcd> ecalDAQTowerStatusToken_;
  edm::ESGetToken<EcalTPGCrystalStatus, EcalTPGCrystalStatusRcd> ecalTPGCrystalStatusToken_;
  edm::ESGetToken<EcalTPGTowerStatus, EcalTPGTowerStatusRcd> ecalTPGTowerStatusToken_;
  edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> ecalIntercalibConstantsToken_;
  edm::ESGetToken<EcalLinearCorrections, EcalLinearCorrectionsRcd> ecalLinearCorrectionsToken_;
  edm::ESGetToken<EcalIntercalibConstantsMC, EcalIntercalibConstantsMCRcd> ecalIntercalibConstantsMCToken_;
  edm::ESGetToken<EcalIntercalibErrors, EcalIntercalibErrorsRcd> ecalIntercalibErrorsToken_;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> ecalGainRatiosToken_;
  edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> ecalWeightXtalGroupsToken_;
  edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> ecalTBWeightsToken_;
  edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> ecalLaserAlphasToken_;
  edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd> ecalLaserAPDPNRatiosToken_;
  edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> ecalLaserAPDPNRatiosRefToken_;
  edm::ESGetToken<EcalClusterCrackCorrParameters, EcalClusterCrackCorrParametersRcd>
      ecalClusterCrackCorrParametersToken_;
  edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> ecalPFRecHitThresholdsToken_;
  edm::ESGetToken<EcalClusterEnergyUncertaintyParameters, EcalClusterEnergyUncertaintyParametersRcd>
      ecalClusterEnergyUncertaintyParametersToken_;
  edm::ESGetToken<EcalClusterEnergyCorrectionParameters, EcalClusterEnergyCorrectionParametersRcd>
      ecalClusterEnergyCorrectionParametersToken_;
  edm::ESGetToken<EcalClusterEnergyCorrectionObjectSpecificParameters,
                  EcalClusterEnergyCorrectionObjectSpecificParametersRcd>
      ecalClusterEnergyCorrectionObjectSpecificParametersToken_;
  edm::ESGetToken<EcalClusterLocalContCorrParameters, EcalClusterLocalContCorrParametersRcd>
      ecalClusterLocalContCorrParametersToken_;
  edm::ESGetToken<Alignments, EBAlignmentRcd> ebAlignmentToken_;
  edm::ESGetToken<Alignments, EEAlignmentRcd> eeAlignmentToken_;
  edm::ESGetToken<Alignments, ESAlignmentRcd> esAlignmentToken_;
  edm::ESGetToken<EcalTimeOffsetConstant, EcalTimeOffsetConstantRcd> ecalTimeOffsetConstantToken_;
  edm::ESGetToken<EcalSampleMask, EcalSampleMaskRcd> ecalSampleMaskToken_;
  edm::ESGetToken<EcalSimComponentShape, EcalSimComponentShapeRcd> ecalSimComponentShapeToken_;
  edm::ESGetToken<EcalSimPulseShape, EcalSimPulseShapeRcd> ecalSimPulseShapeToken_;
  edm::ESGetToken<EcalTimeBiasCorrections, EcalTimeBiasCorrectionsRcd> ecalTimeBiasCorrectionsToken_;
  edm::ESGetToken<EcalSamplesCorrelation, EcalSamplesCorrelationRcd> ecalSamplesCorrelationToken_;
};

#endif
