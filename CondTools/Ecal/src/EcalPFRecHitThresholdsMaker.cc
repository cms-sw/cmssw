#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CondTools/Ecal/interface/EcalPFRecHitThresholdsMaker.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterCrackCorrParametersRcd.h"

#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyUncertaintyParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterEnergyCorrectionObjectSpecificParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
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

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include <vector>

EcalPFRecHitThresholdsMaker::EcalPFRecHitThresholdsMaker(const edm::ParameterSet& iConfig)
    : m_timetype(iConfig.getParameter<std::string>("timetype")) {
  std::string container;
  std::string tag;
  std::string record;

  m_nsigma = iConfig.getParameter<double>("NSigma");
}

EcalPFRecHitThresholdsMaker::~EcalPFRecHitThresholdsMaker() {}

void EcalPFRecHitThresholdsMaker::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if (!dbOutput.isAvailable()) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  edm::ESHandle<EcalPedestals> handle1;
  evtSetup.get<EcalPedestalsRcd>().get(handle1);
  const EcalPedestals* ped_db = handle1.product();
  std::cout << "ped pointer is: " << ped_db << std::endl;

  edm::ESHandle<EcalADCToGeVConstant> handle2;
  evtSetup.get<EcalADCToGeVConstantRcd>().get(handle2);
  const EcalADCToGeVConstant* adc_db = handle2.product();
  std::cout << "adc pointer is: " << adc_db << std::endl;

  edm::ESHandle<EcalIntercalibConstants> handle3;
  evtSetup.get<EcalIntercalibConstantsRcd>().get(handle3);
  const EcalIntercalibConstants* ical_db = handle3.product();
  std::cout << "inter pointer is: " << ical_db << std::endl;

  edm::ESHandle<EcalLaserDbService> laser;
  evtSetup.get<EcalLaserDbRecord>().get(laser);

  EcalPFRecHitThresholds* pfthresh = new EcalPFRecHitThresholds();

  //    const EcalIntercalibConstantMap& icalMap = ical_db->getMap();

  float adc_EB = float(adc_db->getEEValue());
  float adc_EE = float(adc_db->getEBValue());

  //edm::Timestamp tsince;

  for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
    if (iEta == 0)
      continue;
    for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
      // make an EBDetId since we need EBDetId::rawId() to be used as the key for the pedestals
      if (EBDetId::validDetId(iEta, iPhi)) {
        EBDetId ebdetid(iEta, iPhi, EBDetId::ETAPHIMODE);
        EcalPedestals::const_iterator it = ped_db->find(ebdetid.rawId());
        EcalPedestals::Item aped = (*it);

        EcalIntercalibConstants::const_iterator itc = ical_db->find(ebdetid.rawId());
        float calib = (*itc);

        // get laser coefficient
        float lasercalib = 1.;
        lasercalib = laser->getLaserCorrection(ebdetid, evt.time());  // TODO correct time

        EcalPFRecHitThreshold thresh = aped.rms_x12 * calib * adc_EB * lasercalib * m_nsigma;

        if (iPhi == 100)
          std::cout << "Thresh(GeV)=" << thresh << std::endl;

        pfthresh->insert(std::make_pair(ebdetid.rawId(), thresh));
      }
    }
  }

  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX, iY, 1)) {
        EEDetId eedetid(iX, iY, 1);

        EcalPedestals::const_iterator it = ped_db->find(eedetid.rawId());
        EcalPedestals::Item aped = (*it);

        EcalIntercalibConstants::const_iterator itc = ical_db->find(eedetid.rawId());
        float calib = (*itc);

        // get laser coefficient
        float lasercalib = 1.;
        lasercalib = laser->getLaserCorrection(eedetid, evt.time());  // TODO correct time

        EcalPFRecHitThreshold thresh = aped.rms_x12 * calib * adc_EE * lasercalib * m_nsigma;
        pfthresh->insert(std::make_pair(eedetid.rawId(), thresh));
      }
      if (EEDetId::validDetId(iX, iY, -1)) {
        EEDetId eedetid(iX, iY, -1);

        EcalPedestals::const_iterator it = ped_db->find(eedetid.rawId());
        EcalPedestals::Item aped = (*it);

        EcalIntercalibConstants::const_iterator itc = ical_db->find(eedetid.rawId());
        float calib = (*itc);

        // get laser coefficient
        float lasercalib = 1.;
        lasercalib = laser->getLaserCorrection(eedetid, evt.time());  // TODO correct time

        EcalPFRecHitThreshold thresh = aped.rms_x12 * calib * adc_EE * lasercalib * m_nsigma;
        pfthresh->insert(std::make_pair(eedetid.rawId(), thresh));

        if (iX == 50)
          std::cout << "Thresh(GeV)=" << thresh << std::endl;
      }
    }
  }

  dbOutput->createNewIOV<const EcalPFRecHitThresholds>(
      pfthresh, dbOutput->beginOfTime(), dbOutput->endOfTime(), "EcalPFRecHitThresholdsRcd");

  std::cout << "EcalPFRecHitThresholdsMaker wrote it " << std::endl;
}
