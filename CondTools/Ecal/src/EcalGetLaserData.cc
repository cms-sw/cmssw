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
// $Id: EcalGetLaserData.cc,v 1.6 2010/10/18 22:04:26 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "CondTools/Ecal/interface/EcalGetLaserData.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalGetLaserData::EcalGetLaserData(const edm::ParameterSet& iConfig) :
  // m_timetype(iConfig.getParameter<std::string>("timetype")),
  m_cacheIDs(),
  m_records()
{
  std::string container;
  std::string tag;
  std::string record;

  //m_firstRun=(unsigned long long)atoi( iConfig.getParameter<std::string>("firstRun").c_str());
  //m_lastRun=(unsigned long long)atoi( iConfig.getParameter<std::string>("lastRun").c_str());

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toGet = iConfig.getParameter<Parameters>("toGet");
  for(Parameters::iterator i = toGet.begin(); i != toGet.end(); ++i) {
    container = i->getParameter<std::string>("container");
    record = i->getParameter<std::string>("record");
    m_cacheIDs.insert( std::make_pair(container, 0) );
    m_records.insert( std::make_pair(container, record) );

  } //now do what ever initialization is needed

}


EcalGetLaserData::~EcalGetLaserData()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalGetLaserData::analyze(const edm::Event& evt, const edm::EventSetup& evtSetup)
{
  using namespace edm;
  
  // loop on offline DB conditions to be transferred as from config file 
  std::string container;
  std::string record;
  typedef std::map<std::string, std::string>::const_iterator recordIter;
  for (recordIter i = m_records.begin(); i != m_records.end(); ++i) {
    container = (*i).first;
    record = (*i).second;
    
    std::string recordName = m_records[container];

    
    if (container == "EcalLaserAPDPNRatios") {
      
      // get from offline DB the last valid laser set 
      edm::ESHandle<EcalLaserAPDPNRatios> handle;
      evtSetup.get<EcalLaserAPDPNRatiosRcd>().get(handle);

      // this is the offline object 
      EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
      EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
            
      const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = handle.product()->getLaserMap(); 
      const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = handle.product()->getTimeMap(); 

      // loop through ecal barrel
      for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	if(iEta==0) continue;
	for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	  
	  EBDetId ebdetid(iEta,iPhi);
	  int hi = ebdetid.hashedIndex();
	  
	  if (hi<static_cast<int>(laserRatiosMap.size())) {
	    apdpnpair = laserRatiosMap[hi];
	    std::cout << "A sample value of APDPN pair EB : " 
		      << hi << " : " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;	  	  
	  } else {
	    edm::LogError("EcalGetLaserData") << "error with laserRatiosMap!" << std::endl;     
	  }
	  
	}
      }  

      // loop through ecal endcap      
      for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {

	  if (!EEDetId::validDetId(iX,iY,1))
	    continue;
	  
	  EEDetId eedetidpos(iX,iY,1);
	  int hi = eedetidpos.hashedIndex();
	  
	  if (hi< static_cast<int>(laserRatiosMap.size())) {
	    apdpnpair = laserRatiosMap[hi];
	    std::cout << "A sample value of APDPN pair EE+ : " 
		      << hi << " : " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;
	  } else {
	    edm::LogError("EcalGetLaserData") << "error with laserRatiosMap!" << std::endl;     
	  }
	  
	  if (!EEDetId::validDetId(iX,iY,-1))
	    continue;
	  EEDetId eedetidneg(iX,iY,1);
	  hi = eedetidneg.hashedIndex();
	  
	  if (hi< static_cast<int>(laserRatiosMap.size())) {
	    apdpnpair = laserRatiosMap[hi];
	    std::cout << "A sample value of APDPN pair EE- : " 
		      << hi << " : " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;
	  } else {
	    edm::LogError("EcalGetLaserData") << "error with laserRatiosMap!" << std::endl;     
	  }
	}
      }
      
      for(int i=0; i<92; i++){
	timestamp = laserTimeMap[i];  
	std::cout << "A value of timestamp pair : "  
		  << i << " " << timestamp.t1.value() << " , " << timestamp.t2.value() << std::endl;	
      }
      
      std::cout <<".. just retrieved the last valid record from DB "<< std::endl;

    } else if(container == "EcalLaserAPDPNRatiosRef") { 

      // get from offline DB the last valid laser set 
      edm::ESHandle<EcalLaserAPDPNRatiosRef> handle;
      evtSetup.get<EcalLaserAPDPNRatiosRefRcd>().get(handle);

      EcalLaserAPDPNref apdpnref;      
      const EcalLaserAPDPNRatiosRefMap& laserRefMap = handle.product()->getMap(); 
      
      // first barrel
      for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	if(iEta==0) continue;
	for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	  
	  EBDetId ebdetid(iEta,iPhi);
          int hi = ebdetid.hashedIndex();
	  
	  if (hi< static_cast<int>(laserRefMap.size())) {
	    apdpnref = laserRefMap[hi];
	    std::cout << "A sample value of APDPN Reference value EB : "  
		 << hi << " : " << apdpnref << std::endl;	  	  
	  } else { 
	    edm::LogError("EcalGetLaserData") << "error with laserRefMap!" << std::endl;     
	  }	  	  
	}
      }
      
      // now for endcap
      for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {

	  if (!EEDetId::validDetId(iX,iY,1))
	    continue;
	  
	  EEDetId eedetidpos(iX,iY,1);
	  int hi = eedetidpos.hashedIndex();
	  
	  if (hi< static_cast<int>(laserRefMap.size())) {
	    apdpnref = laserRefMap[hi];
	    std::cout << "A sample value of APDPN Reference value EE+ : "  
		 << hi << " : " << apdpnref << std::endl;	  	  		  
	    
	  } else { 
	    edm::LogError("EcalGetLaserData") << "error with laserRefMap!" << std::endl;     
	  }
	  
	  if (!EEDetId::validDetId(iX,iY,-1))
	    continue;
	  EEDetId eedetidneg(iX,iY,-1);
	  hi = eedetidneg.hashedIndex();
	  
	  if (hi< static_cast<int>(laserRefMap.size())) {
	    apdpnref = laserRefMap[hi];
	    std::cout << "A sample value of APDPN Reference value EE- : "  
		 << hi << " : " << apdpnref << std::endl;	  	 
	  } else { 
	    edm::LogError("EcalGetLaserData") << "error with laserRefMap!" << std::endl;     
	  }	      
	}
      }
      
      std::cout << "... just retrieved the last valid record from DB "<< std::endl;
      
    } else if (container == "EcalLaserAlphas") { 

      // get from offline DB the last valid laser set 
      edm::ESHandle<EcalLaserAlphas> handle;
      evtSetup.get<EcalLaserAlphasRcd>().get(handle);

      // this is the offline object 
      EcalLaserAlpha alpha;     
      const EcalLaserAlphaMap& laserAlphaMap = handle.product()->getMap(); // map of apdpns

      // first barrel
      for(int iEta=-EBDetId::MAX_IETA; iEta<=EBDetId::MAX_IETA ;++iEta) {
	if(iEta==0) continue;
	for(int iPhi=EBDetId::MIN_IPHI; iPhi<=EBDetId::MAX_IPHI; ++iPhi) {
	  
	  EBDetId ebdetid(iEta,iPhi);
          int hi = ebdetid.hashedIndex();
	  
	  if (hi< static_cast<int>(laserAlphaMap.size())) {
	    alpha = laserAlphaMap[hi];
	    std::cout << " A sample value of Alpha value EB : " << hi << " : " << alpha << std::endl;
	  } else {
	    edm::LogError("EcalGetLaserData") << "error with laserAlphaMap!" << std::endl;     
	  }	  
	}
      }
      
      // next endcap
      for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
	for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {

	  if (!EEDetId::validDetId(iX,iY,1))
	    continue;

	  EEDetId eedetidpos(iX,iY,1);
	  int hi = eedetidpos.hashedIndex();
	  
	  if (hi< static_cast<int>(laserAlphaMap.size())) {
	    alpha = laserAlphaMap[hi];
	    std::cout << " A sample value of Alpha value EE+ : " << hi << " : " << alpha << std::endl;  
	  } else {
	    edm::LogError("EcalGetLaserData") << "error with laserAlphaMap!" << std::endl;     
	  }	  	      
	  

	  if (!EEDetId::validDetId(iX,iY,-1))
	    continue;
	  EEDetId eedetidneg(iX,iY,-1);
	  hi = eedetidneg.hashedIndex();
	  
	  if (hi< static_cast<int>(laserAlphaMap.size())) {
	    alpha = laserAlphaMap[hi];
	    std::cout << " A sample value of Alpha value EE- : " << hi << " : " << alpha << std::endl;
	  } else {
	    edm::LogError("EcalGetLaserData") << "error with laserAlphaMap!" << std::endl;     
	  }	  
	}
      }
      
      std::cout <<"... just retrieved the last valid record from DB "<< std::endl;

    } else {
      edm::LogError("EcalGetLaserData") << "Cannot retrieve for container: " 
					<< container << std::endl;           
    }
    
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
EcalGetLaserData::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalGetLaserData::endJob() {
}
