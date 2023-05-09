#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondTools/Ecal/interface/EcalDBCopy.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/AlignmentRecord/interface/EBAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/EEAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/ESAlignmentRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"

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

#include <vector>

EcalDBCopy::EcalDBCopy(const edm::ParameterSet& iConfig)
    : m_timetype(iConfig.getParameter<std::string>("timetype")),
      m_cacheIDs(),
      m_records(),
      ecalPedestalToken_(esConsumes()),
      ecalADCtoGeVToken_(esConsumes()),
      ecalTimeCalibToken_(esConsumes()),
      ecalChannelStatusToken_(esConsumes()),
      ecalDQMChannelStatusToken_(esConsumes()),
      ecalDQMTowerStatusToken_(esConsumes()),
      ecalDCSTowerStatusToken_(esConsumes()),
      ecalDAQTowerStatusToken_(esConsumes()),
      ecalTPGCrystalStatusToken_(esConsumes()),
      ecalTPGTowerStatusToken_(esConsumes()),
      ecalIntercalibConstantsToken_(esConsumes()),
      ecalLinearCorrectionsToken_(esConsumes()),
      ecalIntercalibConstantsMCToken_(esConsumes()),
      ecalIntercalibErrorsToken_(esConsumes()),
      ecalGainRatiosToken_(esConsumes()),
      ecalWeightXtalGroupsToken_(esConsumes()),
      ecalTBWeightsToken_(esConsumes()),
      ecalLaserAlphasToken_(esConsumes()),
      ecalLaserAPDPNRatiosToken_(esConsumes()),
      ecalLaserAPDPNRatiosRefToken_(esConsumes()),
      ecalClusterCrackCorrParametersToken_(esConsumes()),
      ecalPFRecHitThresholdsToken_(esConsumes()),
      ecalClusterEnergyUncertaintyParametersToken_(esConsumes()),
      ecalClusterEnergyCorrectionParametersToken_(esConsumes()),
      ecalClusterEnergyCorrectionObjectSpecificParametersToken_(esConsumes()),
      ecalClusterLocalContCorrParametersToken_(esConsumes()),
      ebAlignmentToken_(esConsumes()),
      eeAlignmentToken_(esConsumes()),
      esAlignmentToken_(esConsumes()),
      ecalTimeOffsetConstantToken_(esConsumes()),
      ecalSampleMaskToken_(esConsumes()),
      ecalSimComponentShapeToken_(esConsumes()),
      ecalSimPulseShapeToken_(esConsumes()),
      ecalTimeBiasCorrectionsToken_(esConsumes()),
      ecalSamplesCorrelationToken_(esConsumes()) {
  std::string container;
  std::string record;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for (auto& iparam : toCopy) {
    container = iparam.getParameter<std::string>("container");
    record = iparam.getParameter<std::string>("record");
    m_cacheIDs.emplace(container, 0);
    m_records.emplace(container, record);
  }
}

EcalDBCopy::~EcalDBCopy() {}

void EcalDBCopy::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  for (const auto& irec : m_records) {
    if (shouldCopy(evtSetup, irec.first)) {
      copyToDB(evtSetup, irec.first);
    }
  }
}

bool EcalDBCopy::shouldCopy(const edm::EventSetup& evtSetup, const std::string& container) {
  unsigned long long cacheID = 0;
  if (container == "EcalPedestals") {
    cacheID = evtSetup.get<EcalPedestalsRcd>().cacheIdentifier();
  } else if (container == "EcalADCToGeVConstant") {
    cacheID = evtSetup.get<EcalADCToGeVConstantRcd>().cacheIdentifier();
  } else if (container == "EcalLinearCorrections") {
    cacheID = evtSetup.get<EcalLinearCorrectionsRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibConstants") {
    cacheID = evtSetup.get<EcalIntercalibConstantsRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibConstantsMC") {
    cacheID = evtSetup.get<EcalIntercalibConstantsMCRcd>().cacheIdentifier();
  } else if (container == "EcalIntercalibErrors") {
    cacheID = evtSetup.get<EcalIntercalibErrorsRcd>().cacheIdentifier();
  } else if (container == "EcalGainRatios") {
    cacheID = evtSetup.get<EcalGainRatiosRcd>().cacheIdentifier();
  } else if (container == "EcalWeightXtalGroups") {
    cacheID = evtSetup.get<EcalWeightXtalGroupsRcd>().cacheIdentifier();
  } else if (container == "EcalTBWeights") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else if (container == "EcalLaserAPDPNRatios") {
    cacheID = evtSetup.get<EcalLaserAPDPNRatiosRcd>().cacheIdentifier();
  } else if (container == "EcalLaserAPDPNRatiosRef") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else if (container == "EcalLaserAlphas") {
    cacheID = evtSetup.get<EcalTBWeightsRcd>().cacheIdentifier();
  } else if (container == "EcalChannelStatus") {
    cacheID = evtSetup.get<EcalChannelStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDQMTowerStatus") {
    cacheID = evtSetup.get<EcalDQMTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDQMChannelStatus") {
    cacheID = evtSetup.get<EcalDQMChannelStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDCSTowerStatus") {
    cacheID = evtSetup.get<EcalDCSTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalDAQTowerStatus") {
    cacheID = evtSetup.get<EcalDAQTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EcalTimeCalibConstants") {
    cacheID = evtSetup.get<EcalTimeCalibConstantsRcd>().cacheIdentifier();
  } else if (container == "EcalClusterCrackCorrParameters") {
    cacheID = evtSetup.get<EcalClusterCrackCorrParametersRcd>().cacheIdentifier();
  } else if (container == "EcalPFRecHitThresholds") {
    cacheID = evtSetup.get<EcalPFRecHitThresholdsRcd>().cacheIdentifier();
  } else if (container == "EcalClusterEnergyUncertaintyParameters") {
    cacheID = evtSetup.get<EcalClusterEnergyUncertaintyParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterEnergyCorrectionParameters") {
    cacheID = evtSetup.get<EcalClusterEnergyCorrectionParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterEnergyCorrectionObjectSpecificParameters") {
    cacheID = evtSetup.get<EcalClusterEnergyCorrectionObjectSpecificParametersRcd>().cacheIdentifier();
  } else if (container == "EcalClusterLocalContCorrParameters") {
    cacheID = evtSetup.get<EcalClusterLocalContCorrParametersRcd>().cacheIdentifier();
  } else if (container == "EcalTPGCrystalStatus") {
    cacheID = evtSetup.get<EcalTPGCrystalStatusRcd>().cacheIdentifier();
  } else if (container == "EcalTPGTowerStatus") {
    cacheID = evtSetup.get<EcalTPGTowerStatusRcd>().cacheIdentifier();
  } else if (container == "EBAlignment") {
    cacheID = evtSetup.get<EBAlignmentRcd>().cacheIdentifier();
  } else if (container == "EEAlignment") {
    cacheID = evtSetup.get<EEAlignmentRcd>().cacheIdentifier();
  } else if (container == "ESAlignment") {
    cacheID = evtSetup.get<ESAlignmentRcd>().cacheIdentifier();
  } else if (container == "EcalTimeOffsetConstant") {
    cacheID = evtSetup.get<EcalTimeOffsetConstantRcd>().cacheIdentifier();
  } else if (container == "EcalSampleMask") {
    cacheID = evtSetup.get<EcalSampleMaskRcd>().cacheIdentifier();
  } else if (container == "EcalTimeBiasCorrections") {
    cacheID = evtSetup.get<EcalTimeBiasCorrectionsRcd>().cacheIdentifier();
  } else if (container == "EcalSimComponentShape") {
    cacheID = evtSetup.get<EcalSimComponentShapeRcd>().cacheIdentifier();
  } else if (container == "EcalSimPulseShape") {
    cacheID = evtSetup.get<EcalSimPulseShapeRcd>().cacheIdentifier();
  } else if (container == "EcalSamplesCorrelation") {
    cacheID = evtSetup.get<EcalSamplesCorrelationRcd>().cacheIdentifier();
  }

  else {
    throw cms::Exception("Unknown container");
  }

  if (m_cacheIDs[container] == cacheID) {
    return false;
  } else {
    m_cacheIDs[container] = cacheID;
    return true;
  }
}

void EcalDBCopy::copyToDB(const edm::EventSetup& evtSetup, const std::string& container) {
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if (!dbOutput.isAvailable()) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string recordName = m_records[container];

  if (container == "EcalPedestals") {
    const auto& obj = evtSetup.getData(ecalPedestalToken_);
    edm::LogInfo("EcalDBCopy") << "ped pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalPedestals>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalADCToGeVConstant") {
    const auto& obj = evtSetup.getData(ecalADCtoGeVToken_);
    edm::LogInfo("EcalDBCopy") << "adc pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalADCToGeVConstant>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalTimeCalibConstants") {
    const auto& obj = evtSetup.getData(ecalTimeCalibToken_);
    edm::LogInfo("EcalDBCopy") << "adc pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalTimeCalibConstants>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalChannelStatus") {
    const auto& obj = evtSetup.getData(ecalChannelStatusToken_);
    edm::LogInfo("EcalDBCopy") << "channel status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalChannelStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalDQMChannelStatus") {
    const auto& obj = evtSetup.getData(ecalDQMChannelStatusToken_);
    edm::LogInfo("EcalDBCopy") << "DQM channel status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalDQMChannelStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalDQMTowerStatus") {
    const auto& obj = evtSetup.getData(ecalDQMTowerStatusToken_);
    edm::LogInfo("EcalDBCopy") << "DQM Tower status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalDQMTowerStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalDCSTowerStatus") {
    const auto& obj = evtSetup.getData(ecalDCSTowerStatusToken_);
    edm::LogInfo("EcalDBCopy") << "channel status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalDCSTowerStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalDAQTowerStatus") {
    const auto& obj = evtSetup.getData(ecalDAQTowerStatusToken_);
    edm::LogInfo("EcalDBCopy") << "DAQ channel status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalDAQTowerStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalTPGCrystalStatus") {
    const auto& obj = evtSetup.getData(ecalTPGCrystalStatusToken_);
    edm::LogInfo("EcalDBCopy") << "TPG channel status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalTPGCrystalStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalTPGTowerStatus") {
    const auto& obj = evtSetup.getData(ecalTPGTowerStatusToken_);
    edm::LogInfo("EcalDBCopy") << "TPG tower status pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalTPGTowerStatus>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalIntercalibConstants") {
    const auto& obj = evtSetup.getData(ecalIntercalibConstantsToken_);
    edm::LogInfo("EcalDBCopy") << "inter pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalIntercalibConstants>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalLinearCorrections") {
    const auto& obj = evtSetup.getData(ecalLinearCorrectionsToken_);
    edm::LogInfo("EcalDBCopy") << "inter pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalLinearCorrections>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalIntercalibConstantsMC") {
    const auto& obj = evtSetup.getData(ecalIntercalibConstantsMCToken_);
    edm::LogInfo("EcalDBCopy") << "intercalib MC pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalIntercalibConstantsMC>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalIntercalibErrors") {
    const auto& obj = evtSetup.getData(ecalIntercalibErrorsToken_);
    edm::LogInfo("EcalDBCopy") << "inter pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalIntercalibErrors>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalGainRatios") {
    const auto& obj = evtSetup.getData(ecalGainRatiosToken_);
    edm::LogInfo("EcalDBCopy") << "gain pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalGainRatios>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalWeightXtalGroups") {
    const auto& obj = evtSetup.getData(ecalWeightXtalGroupsToken_);
    edm::LogInfo("EcalDBCopy") << "weight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalWeightXtalGroups>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalTBWeights") {
    const auto& obj = evtSetup.getData(ecalTBWeightsToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalTBWeights>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalLaserAlphas") {
    const auto& obj = evtSetup.getData(ecalLaserAlphasToken_);
    edm::LogInfo("EcalDBCopy") << "ecalLaserAlpha pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalLaserAlphas>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalLaserAPDPNRatios") {
    const auto& obj = evtSetup.getData(ecalLaserAPDPNRatiosToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalLaserAPDPNRatios>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalLaserAPDPNRatiosRef") {
    const auto& obj = evtSetup.getData(ecalLaserAPDPNRatiosRefToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalLaserAPDPNRatiosRef>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalClusterCrackCorrParameters") {
    const auto& obj = evtSetup.getData(ecalClusterCrackCorrParametersToken_);
    edm::LogInfo("EcalDBCopy") << "cluster crack pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalClusterCrackCorrParameters>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalPFRecHitThresholds") {
    const auto& obj = evtSetup.getData(ecalPFRecHitThresholdsToken_);
    edm::LogInfo("EcalDBCopy") << "Ecal PF rec hit thresholds pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalPFRecHitThresholds>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalClusterEnergyUncertaintyParameters") {
    const auto& obj = evtSetup.getData(ecalClusterEnergyUncertaintyParametersToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalClusterEnergyUncertaintyParameters>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalClusterEnergyCorrectionParameters") {
    const auto& obj = evtSetup.getData(ecalClusterEnergyCorrectionParametersToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalClusterEnergyCorrectionParameters>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalClusterEnergyCorrectionObjectSpecificParameters") {
    const auto& obj = evtSetup.getData(ecalClusterEnergyCorrectionObjectSpecificParametersToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalClusterEnergyCorrectionObjectSpecificParameters>(
        obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalClusterLocalContCorrParameters") {
    const auto& obj = evtSetup.getData(ecalClusterLocalContCorrParametersToken_);
    edm::LogInfo("EcalDBCopy") << "tbweight pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalClusterLocalContCorrParameters>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EBAlignment") {
    const auto& obj = evtSetup.getData(ebAlignmentToken_);
    edm::LogInfo("EcalDBCopy") << "EB alignment pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const Alignments>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EEAlignment") {
    const auto& obj = evtSetup.getData(eeAlignmentToken_);
    edm::LogInfo("EcalDBCopy") << "EE alignment pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const Alignments>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "ESAlignment") {
    const auto& obj = evtSetup.getData(esAlignmentToken_);
    edm::LogInfo("EcalDBCopy") << "ES alignment pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const Alignments>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalTimeOffsetConstant") {
    const auto& obj = evtSetup.getData(ecalTimeOffsetConstantToken_);
    edm::LogInfo("EcalDBCopy") << "TimeOffset pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalTimeOffsetConstant>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalSampleMask") {
    const auto& obj = evtSetup.getData(ecalSampleMaskToken_);
    edm::LogInfo("EcalDBCopy") << "sample mask pointer is: " << &obj << std::endl;
    dbOutput->createOneIOV<const EcalSampleMask>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalSimComponentShape") {
    const auto& obj = evtSetup.getData(ecalSimComponentShapeToken_);
    dbOutput->createOneIOV<const EcalSimComponentShape>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalSimPulseShape") {
    const auto& obj = evtSetup.getData(ecalSimPulseShapeToken_);
    dbOutput->createOneIOV<const EcalSimPulseShape>(obj, dbOutput->beginOfTime(), recordName);

  } else if (container == "EcalTimeBiasCorrections") {
    const auto& obj = evtSetup.getData(ecalTimeBiasCorrectionsToken_);
    edm::LogInfo("EcalDBCopy") << "TimeBiasCorrections pointer is: " << &obj << std::endl;
    EcalTimeBiasCorrections bias_;
    std::vector<float> vect = obj.EBTimeCorrAmplitudeBins;
    copy(vect.begin(), vect.end(), back_inserter(bias_.EBTimeCorrAmplitudeBins));
    vect = obj.EBTimeCorrShiftBins;
    copy(vect.begin(), vect.end(), back_inserter(bias_.EBTimeCorrShiftBins));
    vect = obj.EETimeCorrAmplitudeBins;
    copy(vect.begin(), vect.end(), back_inserter(bias_.EETimeCorrAmplitudeBins));
    vect = obj.EETimeCorrShiftBins;
    copy(vect.begin(), vect.end(), back_inserter(bias_.EETimeCorrShiftBins));
    dbOutput->writeOneIOV(bias_, dbOutput->beginOfTime(), "EcalTimeBiasCorrectionsRcd");

  } else if (container == "EcalSamplesCorrelation") {
    const auto& obj = evtSetup.getData(ecalSamplesCorrelationToken_);
    edm::LogInfo("EcalDBCopy") << "SamplesCorrelation pointer is: " << &obj << std::endl;
    EcalSamplesCorrelation correl_;
    std::vector<double> vect = obj.EBG12SamplesCorrelation;
    copy(vect.begin(), vect.end(), back_inserter(correl_.EBG12SamplesCorrelation));
    vect = obj.EBG6SamplesCorrelation;
    copy(vect.begin(), vect.end(), back_inserter(correl_.EBG6SamplesCorrelation));
    vect = obj.EBG1SamplesCorrelation;
    copy(vect.begin(), vect.end(), back_inserter(correl_.EBG1SamplesCorrelation));
    vect = obj.EEG12SamplesCorrelation;
    copy(vect.begin(), vect.end(), back_inserter(correl_.EEG12SamplesCorrelation));
    vect = obj.EEG6SamplesCorrelation;
    copy(vect.begin(), vect.end(), back_inserter(correl_.EEG6SamplesCorrelation));
    vect = obj.EEG1SamplesCorrelation;
    copy(vect.begin(), vect.end(), back_inserter(correl_.EEG1SamplesCorrelation));
    dbOutput->writeOneIOV(correl_, dbOutput->beginOfTime(), "EcalSamplesCorrelationRcd");

  } else {
    throw cms::Exception("Unknown container");
  }

  edm::LogInfo("EcalDBCopy") << "EcalDBCopy wrote " << recordName << std::endl;
}
