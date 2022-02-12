// -*- C++ -*-
//
// Package:    Ecal
// Class:      GetLaserData
//
/**\class GetLaserData GetLaserData.cc CondTools/Ecal/src/EcalGetLaserData.cc

 Description: Gets Ecal Laser values from DB

*/
//
// Original Author:  Vladlen Timciuc
//         Created:  Wed Jul  4 13:55:56 CEST 2007
// $Id: EcalGetLaserData.cc,v 1.5 2009/12/21 14:22:03 ebecheva Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondTools/Ecal/interface/EcalGetLaserData.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalGetLaserData::EcalGetLaserData(const edm::ParameterSet& iConfig)
    :  // m_timetype(iConfig.getParameter<std::string>("timetype")),
      m_cacheIDs(),
      m_records(),
      ecalLaserAPDPNRatiosToken_(esConsumes()),
      ecalLaserAPDPNRatiosRefToken_(esConsumes()),
      ecalLaserAlphasToken_(esConsumes()) {
  //m_firstRun=(unsigned long long)atoi( iConfig.getParameter<std::string>("firstRun").c_str());
  //m_lastRun=(unsigned long long)atoi( iConfig.getParameter<std::string>("lastRun").c_str());
  std::string container;
  std::string record;
  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters toGet = iConfig.getParameter<Parameters>("toGet");
  for (const auto& iparam : toGet) {
    container = iparam.getParameter<std::string>("container");
    record = iparam.getParameter<std::string>("record");
    m_cacheIDs.emplace(container, 0);
    m_records.emplace(container, record);

  }  //now do what ever initialization is needed
}

EcalGetLaserData::~EcalGetLaserData() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void EcalGetLaserData::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) {
  using namespace edm;

  // loop on offline DB conditions to be transferred as from config file
  for (const auto& irec : m_records) {
    const std::string& container = irec.first;
    //record = irec.second;

    if (container == "EcalLaserAPDPNRatios") {
      // get from offline DB the last valid laser set
      const EcalLaserAPDPNRatios* laserapdpnrRatios = &evtSetup.getData(ecalLaserAPDPNRatiosToken_);
      // this is the offline object
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;

      const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = laserapdpnrRatios->getLaserMap();
      const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = laserapdpnrRatios->getTimeMap();

      // loop through ecal barrel
      for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
        if (iEta == 0)
          continue;
        for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
          EBDetId ebdetid(iEta, iPhi);
          int hi = ebdetid.hashedIndex();

          if (hi < static_cast<int>(laserRatiosMap.size())) {
            apdpnpair = laserRatiosMap[hi];
            edm::LogInfo("EcalGetLaserData") << "A sample value of APDPN pair EB : " << hi << " : " << apdpnpair.p1
                                             << " , " << apdpnpair.p2 << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserRatiosMap!" << std::endl;
          }
        }
      }

      // loop through ecal endcap
      for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
        for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
          if (!EEDetId::validDetId(iX, iY, 1))
            continue;

          EEDetId eedetidpos(iX, iY, 1);
          int hi = eedetidpos.hashedIndex();

          if (hi < static_cast<int>(laserRatiosMap.size())) {
            apdpnpair = laserRatiosMap[hi];
            edm::LogInfo("EcalGetLaserData") << "A sample value of APDPN pair EE+ : " << hi << " : " << apdpnpair.p1
                                             << " , " << apdpnpair.p2 << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserRatiosMap!" << std::endl;
          }

          if (!EEDetId::validDetId(iX, iY, -1))
            continue;
          EEDetId eedetidneg(iX, iY, 1);
          hi = eedetidneg.hashedIndex();

          if (hi < static_cast<int>(laserRatiosMap.size())) {
            apdpnpair = laserRatiosMap[hi];
            edm::LogInfo("EcalGetLaserData") << "A sample value of APDPN pair EE- : " << hi << " : " << apdpnpair.p1
                                             << " , " << apdpnpair.p2 << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserRatiosMap!" << std::endl;
          }
        }
      }

      for (int i = 0; i < 92; i++) {
        timestamp = laserTimeMap[i];
        edm::LogInfo("EcalGetLaserData") << "A value of timestamp pair : " << i << " " << timestamp.t1.value() << " , "
                                         << timestamp.t2.value() << std::endl;
      }

      edm::LogInfo("EcalGetLaserData") << ".. just retrieved the last valid record from DB " << std::endl;

    } else if (container == "EcalLaserAPDPNRatiosRef") {
      // get from offline DB the last valid laser set
      EcalLaserAPDPNref apdpnref;
      const EcalLaserAPDPNRatiosRefMap& laserRefMap = (&evtSetup.getData(ecalLaserAPDPNRatiosRefToken_))->getMap();

      // first barrel
      for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
        if (iEta == 0)
          continue;
        for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
          EBDetId ebdetid(iEta, iPhi);
          int hi = ebdetid.hashedIndex();

          if (hi < static_cast<int>(laserRefMap.size())) {
            apdpnref = laserRefMap[hi];
            edm::LogInfo("EcalGetLaserData")
                << "A sample value of APDPN Reference value EB : " << hi << " : " << apdpnref << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserRefMap!" << std::endl;
          }
        }
      }

      // now for endcap
      for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
        for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
          if (!EEDetId::validDetId(iX, iY, 1))
            continue;

          EEDetId eedetidpos(iX, iY, 1);
          int hi = eedetidpos.hashedIndex();

          if (hi < static_cast<int>(laserRefMap.size())) {
            apdpnref = laserRefMap[hi];
            edm::LogInfo("EcalGetLaserData")
                << "A sample value of APDPN Reference value EE+ : " << hi << " : " << apdpnref << std::endl;

          } else {
            edm::LogError("EcalGetLaserData") << "error with laserRefMap!" << std::endl;
          }

          if (!EEDetId::validDetId(iX, iY, -1))
            continue;
          EEDetId eedetidneg(iX, iY, -1);
          hi = eedetidneg.hashedIndex();

          if (hi < static_cast<int>(laserRefMap.size())) {
            apdpnref = laserRefMap[hi];
            edm::LogInfo("EcalGetLaserData")
                << "A sample value of APDPN Reference value EE- : " << hi << " : " << apdpnref << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserRefMap!" << std::endl;
          }
        }
      }

      edm::LogInfo("EcalGetLaserData") << "... just retrieved the last valid record from DB " << std::endl;

    } else if (container == "EcalLaserAlphas") {
      // get from offline DB the last valid laser set
      // this is the offline object
      EcalLaserAlpha alpha;
      const EcalLaserAlphaMap& laserAlphaMap = (&evtSetup.getData(ecalLaserAlphasToken_))->getMap();  // map of apdpns

      // first barrel
      for (int iEta = -EBDetId::MAX_IETA; iEta <= EBDetId::MAX_IETA; ++iEta) {
        if (iEta == 0)
          continue;
        for (int iPhi = EBDetId::MIN_IPHI; iPhi <= EBDetId::MAX_IPHI; ++iPhi) {
          EBDetId ebdetid(iEta, iPhi);
          int hi = ebdetid.hashedIndex();

          if (hi < static_cast<int>(laserAlphaMap.size())) {
            alpha = laserAlphaMap[hi];
            edm::LogInfo("EcalGetLaserData")
                << " A sample value of Alpha value EB : " << hi << " : " << alpha << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserAlphaMap!" << std::endl;
          }
        }
      }

      // next endcap
      for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
        for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
          if (!EEDetId::validDetId(iX, iY, 1))
            continue;

          EEDetId eedetidpos(iX, iY, 1);
          int hi = eedetidpos.hashedIndex();

          if (hi < static_cast<int>(laserAlphaMap.size())) {
            alpha = laserAlphaMap[hi];
            edm::LogInfo("EcalGetLaserData")
                << " A sample value of Alpha value EE+ : " << hi << " : " << alpha << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserAlphaMap!" << std::endl;
          }

          if (!EEDetId::validDetId(iX, iY, -1))
            continue;
          EEDetId eedetidneg(iX, iY, -1);
          hi = eedetidneg.hashedIndex();

          if (hi < static_cast<int>(laserAlphaMap.size())) {
            alpha = laserAlphaMap[hi];
            edm::LogInfo("EcalGetLaserData")
                << " A sample value of Alpha value EE- : " << hi << " : " << alpha << std::endl;
          } else {
            edm::LogError("EcalGetLaserData") << "error with laserAlphaMap!" << std::endl;
          }
        }
      }

      edm::LogInfo("EcalGetLaserData") << "... just retrieved the last valid record from DB " << std::endl;

    } else {
      edm::LogError("EcalGetLaserData") << "Cannot retrieve for container: " << container << std::endl;
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void EcalGetLaserData::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void EcalGetLaserData::endJob() {}
