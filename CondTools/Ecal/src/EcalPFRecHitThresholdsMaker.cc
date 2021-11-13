#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CondTools/Ecal/interface/EcalPFRecHitThresholdsMaker.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include <vector>

EcalPFRecHitThresholdsMaker::EcalPFRecHitThresholdsMaker(const edm::ParameterSet& iConfig)
    : m_timetype(iConfig.getParameter<std::string>("timetype")),
      ecalPedestalsToken_(esConsumes()),
      ecalADCToGeVConstantToken_(esConsumes()),
      ecalIntercalibConstantsToken_(esConsumes()),
      ecalLaserDbServiceToken_(esConsumes()) {
  m_nsigma = iConfig.getParameter<double>("NSigma");
}

EcalPFRecHitThresholdsMaker::~EcalPFRecHitThresholdsMaker() {}

void EcalPFRecHitThresholdsMaker::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if (!dbOutput.isAvailable()) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  const EcalPedestals* ped_db = &evtSetup.getData(ecalPedestalsToken_);
  edm::LogInfo("EcalPFRecHitThresholdsMaker") << "ped pointer is: " << ped_db << std::endl;

  const EcalADCToGeVConstant* adc_db = &evtSetup.getData(ecalADCToGeVConstantToken_);
  edm::LogInfo("EcalPFRecHitThresholdsMaker") << "adc pointer is: " << adc_db << std::endl;

  const EcalIntercalibConstants* ical_db = &evtSetup.getData(ecalIntercalibConstantsToken_);
  edm::LogInfo("EcalPFRecHitThresholdsMaker") << "inter pointer is: " << ical_db << std::endl;

  const auto laser = evtSetup.getHandle(ecalLaserDbServiceToken_);

  EcalPFRecHitThresholds pfthresh;

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
          edm::LogInfo("EcalPFRecHitThresholdsMaker") << "Thresh(GeV)=" << thresh << std::endl;

        pfthresh.insert(std::make_pair(ebdetid.rawId(), thresh));
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
        pfthresh.insert(std::make_pair(eedetid.rawId(), thresh));
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
        pfthresh.insert(std::make_pair(eedetid.rawId(), thresh));

        if (iX == 50)
          edm::LogInfo("EcalPFRecHitThresholdsMaker") << "Thresh(GeV)=" << thresh << std::endl;
      }
    }
  }

  dbOutput->createOneIOV<const EcalPFRecHitThresholds>(pfthresh, dbOutput->beginOfTime(), "EcalPFRecHitThresholdsRcd");

  edm::LogInfo("EcalPFRecHitThresholdsMaker") << "EcalPFRecHitThresholdsMaker wrote it " << std::endl;
}
