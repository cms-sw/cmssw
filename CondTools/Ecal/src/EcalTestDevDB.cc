#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "CondTools/Ecal/interface/EcalTestDevDB.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

using namespace std;

EcalTestDevDB::EcalTestDevDB(const edm::ParameterSet& iConfig)
    : m_timetype(iConfig.getParameter<std::string>("timetype")), m_cacheIDs(), m_records() {
  std::string container;
  std::string tag;
  std::string record;

  m_firstRun = static_cast<unsigned long>(atoi(iConfig.getParameter<std::string>("firstRun").c_str()));
  m_lastRun = static_cast<unsigned long>(atoi(iConfig.getParameter<std::string>("lastRun").c_str()));
  m_interval = static_cast<unsigned long>(atoi(iConfig.getParameter<std::string>("interval").c_str()));

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toCopy = iConfig.getParameter<Parameters>("toCopy");
  for (Parameters::iterator i = toCopy.begin(); i != toCopy.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert(std::make_pair(container, 0));
    m_records.insert(std::make_pair(container, record));
  }
}

EcalTestDevDB::~EcalTestDevDB() {}

void EcalTestDevDB::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  edm::Service<cond::service::PoolDBOutputService> dbOutput;
  if (!dbOutput.isAvailable()) {
    throw cms::Exception("PoolDBOutputService is not available");
  }

  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;

    std::string recordName = m_records[container];

    // Loop through each of the runs

    unsigned long nrec = (m_lastRun - m_firstRun) / m_interval + 1;
    unsigned long nstart = 0;
    if (m_firstRun == 0 && m_lastRun == 0) {
      // it should do at least once the loop
      nstart = 0;
      nrec = 1;
    }

    for (unsigned long i = nstart; i < nrec; i++) {
      unsigned long irun = m_firstRun + i * m_interval;

      // Arguments 0 0 mean infinite IOV
      if (m_firstRun == 0 && m_lastRun == 0) {
        edm::LogInfo("EcalTestDevDB") << "Infinite IOV mode";
        irun = edm::IOVSyncValue::endOfTime().eventID().run();
      }

      edm::LogInfo("EcalTestDevDB") << "Starting Transaction for run " << irun << "...";

      if (container == "EcalPedestals") {
        const auto condObject = generateEcalPedestals();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalPedestals>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalPedestals>(*condObject, irun, recordName);
        }

      } else if (container == "EcalADCToGeVConstant") {
        const auto condObject = generateEcalADCToGeVConstant();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalADCToGeVConstant>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalADCToGeVConstant>(*condObject, irun, recordName);
        }

      } else if (container == "EcalIntercalibConstants") {
        const auto condObject = generateEcalIntercalibConstants();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalIntercalibConstants>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalIntercalibConstants>(*condObject, irun, recordName);
        }
      } else if (container == "EcalLinearCorrections") {
        const auto condObject = generateEcalLinearCorrections();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalLinearCorrections>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalLinearCorrections>(*condObject, irun, recordName);
        }

      } else if (container == "EcalGainRatios") {
        const auto condObject = generateEcalGainRatios();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalGainRatios>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalGainRatios>(*condObject, irun, recordName);
        }

      } else if (container == "EcalWeightXtalGroups") {
        const auto condObject = generateEcalWeightXtalGroups();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalWeightXtalGroups>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalWeightXtalGroups>(*condObject, irun, recordName);
        }

      } else if (container == "EcalTBWeights") {
        const auto condObject = generateEcalTBWeights();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalTBWeights>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalTBWeights>(*condObject, irun, recordName);
        }

      } else if (container == "EcalLaserAPDPNRatios") {
        const auto condObject = generateEcalLaserAPDPNRatios(irun);
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalLaserAPDPNRatios>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalLaserAPDPNRatios>(*condObject, irun, recordName);
        }
      } else if (container == "EcalLaserAPDPNRatiosRef") {
        const auto condObject = generateEcalLaserAPDPNRatiosRef();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalLaserAPDPNRatiosRef>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalLaserAPDPNRatiosRef>(*condObject, irun, recordName);
        }
      } else if (container == "EcalLaserAlphas") {
        const auto condObject = generateEcalLaserAlphas();
        if (irun == m_firstRun && dbOutput->isNewTagRequest(recordName)) {
          // create new
          edm::LogInfo("EcalTestDevDB") << "First One ";
          dbOutput->createOneIOV<const EcalLaserAlphas>(*condObject, dbOutput->beginOfTime(), recordName);
        } else {
          // append
          edm::LogInfo("EcalTestDevDB") << "Old One ";
          dbOutput->appendOneIOV<const EcalLaserAlphas>(*condObject, irun, recordName);
        }
      } else {
        edm::LogWarning("EcalTestDevDB") << "it does not work yet for " << container << "...";
      }
    }
  }
}

//-------------------------------------------------------------
std::shared_ptr<EcalPedestals> EcalTestDevDB::generateEcalPedestals() {
  //-------------------------------------------------------------

  auto peds = std::make_shared<EcalPedestals>();
  EcalPedestals::Item item;
  for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
    if (iEta == 0)
      continue;
    for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
      item.mean_x1 = 200. * ((double)std::rand() / (double(RAND_MAX) + double(1)));
      item.rms_x1 = (double)std::rand() / (double(RAND_MAX) + double(1));
      item.mean_x6 = 1200. * ((double)std::rand() / (double(RAND_MAX) + double(1)));
      item.rms_x6 = 6. * ((double)std::rand() / (double(RAND_MAX) + double(1)));
      item.mean_x12 = 2400. * ((double)std::rand() / (double(RAND_MAX) + double(1)));
      item.rms_x12 = 12. * ((double)std::rand() / (double(RAND_MAX) + double(1)));

      EBDetId ebdetid(iEta, iPhi);
      peds->insert(std::make_pair(ebdetid.rawId(), item));
    }
  }
  return peds;
}

//-------------------------------------------------------------
std::shared_ptr<EcalADCToGeVConstant> EcalTestDevDB::generateEcalADCToGeVConstant() {
  //-------------------------------------------------------------

  double r = (double)std::rand() / (double(RAND_MAX) + double(1));
  auto agc = std::make_shared<EcalADCToGeVConstant>(36. + r * 4., 60. + r * 4);
  return agc;
}

//-------------------------------------------------------------
std::shared_ptr<EcalIntercalibConstants> EcalTestDevDB::generateEcalIntercalibConstants() {
  //-------------------------------------------------------------

  auto ical = std::make_shared<EcalIntercalibConstants>();

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta, iphi);

      double r = (double)std::rand() / (double(RAND_MAX) + double(1));
      ical->setValue(ebid.rawId(), 0.85 + r * 0.3);
    }  // loop over phi
  }    // loop over eta
  return ical;
}

//-------------------------------------------------------------
std::shared_ptr<EcalLinearCorrections> EcalTestDevDB::generateEcalLinearCorrections() {
  //-------------------------------------------------------------

  auto ical = std::make_shared<EcalLinearCorrections>();

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      if (EBDetId::validDetId(ieta, iphi)) {
        EBDetId ebid(ieta, iphi);

        EcalLinearCorrections::Values pairAPDPN;
        pairAPDPN.p1 = 1.0;
        pairAPDPN.p2 = 1.0;
        pairAPDPN.p3 = 1.0;
        ical->setValue(ebid, pairAPDPN);
      }
    }
  }

  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      if (EEDetId::validDetId(iX, iY, 1)) {
        EEDetId eedetidpos(iX, iY, 1);

        EcalLinearCorrections::Values pairAPDPN;
        pairAPDPN.p1 = 1.0;
        pairAPDPN.p2 = 1.0;
        pairAPDPN.p3 = 1.0;

        ical->setValue(eedetidpos, pairAPDPN);
      }

      if (EEDetId::validDetId(iX, iY, -1)) {
        EEDetId eedetidneg(iX, iY, -1);

        EcalLinearCorrections::Values pairAPDPN;
        pairAPDPN.p1 = 1.0;
        pairAPDPN.p2 = 1.0;
        pairAPDPN.p3 = 1.0;

        ical->setValue(eedetidneg, pairAPDPN);
      }
    }
  }

  EcalLinearCorrections::Times TimeStamp;
  for (int i = 0; i < 92; i++) {
    TimeStamp.t1 = edm::Timestamp(0);
    TimeStamp.t2 = edm::Timestamp(edm::Timestamp::endOfTime().value());
    TimeStamp.t3 = edm::Timestamp(edm::Timestamp::endOfTime().value());

    ical->setTime(i, TimeStamp);
  }

  return ical;
}

//-------------------------------------------------------------
std::shared_ptr<EcalGainRatios> EcalTestDevDB::generateEcalGainRatios() {
  //-------------------------------------------------------------

  // create gain ratios
  auto gratio = std::make_shared<EcalGainRatios>();

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta, iphi);

      double r = (double)std::rand() / (double(RAND_MAX) + double(1));

      EcalMGPAGainRatio gr;
      gr.setGain12Over6(1.9 + r * 0.2);
      gr.setGain6Over1(5.9 + r * 0.2);

      gratio->setValue(ebid.rawId(), gr);

    }  // loop over phi
  }    // loop over eta
  return gratio;
}

//-------------------------------------------------------------
std::shared_ptr<EcalWeightXtalGroups> EcalTestDevDB::generateEcalWeightXtalGroups() {
  //-------------------------------------------------------------

  auto xtalGroups = std::make_shared<EcalWeightXtalGroups>();
  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebid(ieta, iphi);
      xtalGroups->setValue(ebid.rawId(), EcalXtalGroupId(ieta));  // define rings in eta
    }
  }
  return xtalGroups;
}

//-------------------------------------------------------------
std::shared_ptr<EcalTBWeights> EcalTestDevDB::generateEcalTBWeights() {
  //-------------------------------------------------------------

  auto tbwgt = std::make_shared<EcalTBWeights>();

  // create weights for each distinct group ID
  int nMaxTDC = 10;
  for (int igrp = -EBDetId::MAX_IETA; igrp <= EBDetId::MAX_IETA; ++igrp) {
    if (igrp == 0)
      continue;
    for (int itdc = 1; itdc <= nMaxTDC; ++itdc) {
      // generate random number
      double r = (double)std::rand() / (double(RAND_MAX) + double(1));

      // make a new set of weights
      EcalWeightSet wgt;
      EcalWeightSet::EcalWeightMatrix& mat1 = wgt.getWeightsBeforeGainSwitch();
      EcalWeightSet::EcalWeightMatrix& mat2 = wgt.getWeightsAfterGainSwitch();

      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 10; ++j) {
          double ww = igrp * itdc * r + i * 10. + j;
          mat1(i, j) = ww;
          mat2(i, j) = 100 + ww;
        }
      }

      // fill the chi2 matrcies
      r = (double)std::rand() / (double(RAND_MAX) + double(1));
      EcalWeightSet::EcalChi2WeightMatrix& mat3 = wgt.getChi2WeightsBeforeGainSwitch();
      EcalWeightSet::EcalChi2WeightMatrix& mat4 = wgt.getChi2WeightsAfterGainSwitch();
      for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
          double ww = igrp * itdc * r + i * 10. + j;
          mat3(i, j) = 1000 + ww;
          mat4(i, j) = 1000 + 100 + ww;
        }
      }

      // put the weight in the container
      tbwgt->setValue(std::make_pair(igrp, itdc), wgt);
    }
  }
  return tbwgt;
}

//--------------------------------------------------------------
std::shared_ptr<EcalLaserAPDPNRatios> EcalTestDevDB::generateEcalLaserAPDPNRatios(uint32_t i_run) {
  //--------------------------------------------------------------

  auto laser = std::make_shared<EcalLaserAPDPNRatios>();

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair APDPNpair;
  EcalLaserAPDPNRatios::EcalLaserTimeStamp TimeStamp;

  //  if((m_firstRun == 0 && i_run == 0) || (m_firstRun == 1 && i_run == 1)){

  edm::LogInfo("EcalTestDevDB") << "First & last run: " << i_run << " " << m_firstRun << " " << m_lastRun;
  if (m_firstRun == i_run && (i_run == 0 || i_run == 1)) {
    APDPNpair.p1 = (double(1) + 1 / double(log(exp(1) + double((i_run - m_firstRun) * 10)))) / double(2);
    APDPNpair.p2 = (double(1) + 1 / double(log(exp(1) + double((i_run - m_firstRun) * 10) + double(10)))) / double(2);
    APDPNpair.p3 = double(0);
    edm::LogInfo("EcalTestDevDB") << i_run << " " << m_firstRun << " " << APDPNpair.p1 << " " << APDPNpair.p2;

    for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
      if (iEta == 0)
        continue;
      for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
        //APDPNpair.p1= double(1);
        //APDPNpair.p2= double(1);

        EBDetId ebid(iEta, iPhi);
        int hi = ebid.hashedIndex();

        if (hi < static_cast<int>(laser->getLaserMap().size())) {
          laser->setValue(hi, APDPNpair);
        } else {
          edm::LogError("EcalTestDevDB") << "error with laser Map (ratio)!";
          continue;
        }
      }
    }

    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals

        if (!EEDetId::validDetId(iX, iY, 1))
          continue;

        EEDetId eedetidpos(iX, iY, 1);
        //APDPNpair.p1 = double(1);
        //APDPNpair.p2 = double(1);

        int hi = eedetidpos.hashedIndex() + EBDetId::MAX_HASH + 1;
        if (hi < static_cast<int>(laser->getLaserMap().size())) {
          laser->setValue(hi, APDPNpair);
        } else {
          edm::LogError("EcalTestDevDB") << "error with laser Map (ratio)!";
          continue;
        }

        if (!EEDetId::validDetId(iX, iY, -1))
          continue;

        EEDetId eedetidneg(iX, iY, -1);
        //APDPNpair.p1 = double(1);
        //APDPNpair.p2 = double(1);
        hi = eedetidneg.hashedIndex() + EBDetId::MAX_HASH + 1;
        if (hi < static_cast<int>(laser->getLaserMap().size())) {
          laser->setValue(hi, APDPNpair);
        } else {
          edm::LogError("EcalTestDevDB") << "error with laser Map (ratio)!";
          continue;
        }
      }
    }

    for (int i = 0; i < 92; i++) {
      if (i < static_cast<int>(laser->getTimeMap().size())) {
        TimeStamp.t1 = edm::Timestamp(1380 * (i_run - m_firstRun) + 15 * i);
        TimeStamp.t2 = edm::Timestamp(1380 * (i_run - m_firstRun + 1) + 15 * i);
        laser->setTime(i, TimeStamp);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (time)!";
        continue;
      }
    }

  } else {
    APDPNpair.p1 = (double(1) + 1 / double(log(exp(1) + double((i_run - m_firstRun) * 10)))) / double(2);
    APDPNpair.p2 = (double(1) + 1 / double(log(exp(1) + double((i_run - m_firstRun) * 10) + double(10)))) / double(2);
    APDPNpair.p3 = double(0);
    edm::LogInfo("EcalTestDevDB") << i_run << " " << m_firstRun << " " << APDPNpair.p1 << " " << APDPNpair.p2;

    for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
      if (iEta == 0)
        continue;
      for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
        EBDetId ebid(iEta, iPhi);
        int hi = ebid.hashedIndex();

        if (hi < static_cast<int>(laser->getLaserMap().size())) {
          laser->setValue(hi, APDPNpair);
        } else {
          edm::LogError("EcalTestDevDB") << "error with laser Map (ratio)!";
        }
      }
    }
    for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
      for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
        // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals

        if (!EEDetId::validDetId(iX, iY, 1))
          continue;

        EEDetId eedetidpos(iX, iY, 1);
        int hi = eedetidpos.hashedIndex() + EBDetId::MAX_HASH + 1;
        if (hi < static_cast<int>(laser->getLaserMap().size())) {
          laser->setValue(hi, APDPNpair);
        } else {
          edm::LogError("EcalTestDevDB") << "error with laser Map (ratio)!";
          continue;
        }

        if (!EEDetId::validDetId(iX, iY, -1))
          continue;

        EEDetId eedetidneg(iX, iY, -1);
        hi = eedetidneg.hashedIndex() + EBDetId::MAX_HASH + 1;
        if (hi < static_cast<int>(laser->getLaserMap().size())) {
          laser->setValue(hi, APDPNpair);
        } else {
          edm::LogError("EcalTestDevDB") << "error with laser Map (ratio)!";
          continue;
        }
      }
    }

    for (int i = 0; i < 92; i++) {
      if (i < static_cast<int>(laser->getTimeMap().size())) {
        TimeStamp.t1 = edm::Timestamp(1380 * (i_run - m_firstRun) + 15 * i);
        TimeStamp.t2 = edm::Timestamp(1380 * (i_run - m_firstRun + 1) + 15 * i);
        laser->setTime(i, TimeStamp);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (time)!";
        continue;
      }
    }
  }

  return laser;
}

//--------------------------------------------------------------
std::shared_ptr<EcalLaserAPDPNRatiosRef> EcalTestDevDB::generateEcalLaserAPDPNRatiosRef() {
  //--------------------------------------------------------------

  auto laser = std::make_shared<EcalLaserAPDPNRatiosRef>();

  EcalLaserAPDPNref APDPNref;

  for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
    if (iEta == 0)
      continue;
    for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
      APDPNref = double(1.5);
      EBDetId ebid(iEta, iPhi);

      int hi = ebid.hashedIndex();
      if (hi < static_cast<int>(laser->getMap().size())) {
        laser->setValue(hi, APDPNref);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (ref)!";
      }
    }
  }

  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      if (!EEDetId::validDetId(iX, iY, 1))
        continue;

      EEDetId eedetidpos(iX, iY, 1);
      APDPNref = double(1.5);

      int hi = eedetidpos.hashedIndex() + EBDetId::MAX_HASH + 1;
      if (hi < static_cast<int>(laser->getMap().size())) {
        laser->setValue(hi, APDPNref);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (ref)!";
      }

      if (!EEDetId::validDetId(iX, iY, -1))
        continue;

      EEDetId eedetidneg(iX, iY, -1);
      APDPNref = double(1.5);

      hi = eedetidneg.hashedIndex() + EBDetId::MAX_HASH + 1;
      if (hi < static_cast<int>(laser->getMap().size())) {
        laser->setValue(hi, APDPNref);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (ref)!";
      }
    }
  }

  return laser;
}

//--------------------------------------------------------------
std::shared_ptr<EcalLaserAlphas> EcalTestDevDB::generateEcalLaserAlphas() {
  //--------------------------------------------------------------

  auto laser = std::make_shared<EcalLaserAlphas>();

  EcalLaserAlpha Alpha;

  for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
    if (iEta == 0)
      continue;
    for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
      Alpha = double(1.55);
      EBDetId ebid(iEta, iPhi);

      int hi = ebid.hashedIndex();
      if (hi < static_cast<int>(laser->getMap().size())) {
        laser->setValue(hi, Alpha);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (alpha)!";
      }
    }
  }

  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      if (!EEDetId::validDetId(iX, iY, 1))
        continue;

      EEDetId eedetidpos(iX, iY, 1);
      Alpha = double(1.55);

      int hi = eedetidpos.hashedIndex() + EBDetId::MAX_HASH + 1;
      if (hi < static_cast<int>(laser->getMap().size())) {
        laser->setValue(hi, Alpha);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (alpha)!";
      }

      if (!EEDetId::validDetId(iX, iY, -1))
        continue;
      EEDetId eedetidneg(iX, iY, -1);
      Alpha = double(1.55);

      hi = eedetidneg.hashedIndex() + EBDetId::MAX_HASH + 1;
      if (hi < static_cast<int>(laser->getMap().size())) {
        laser->setValue(hi, Alpha);
      } else {
        edm::LogError("EcalTestDevDB") << "error with laser Map (alpha)!";
      }
    }
  }

  return laser;
}
