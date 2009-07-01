//
// Toyoko Orimoto (Caltech), 10 July 2007
//


// system include files
#include <memory>
//#include <time.h>
#include <string>
#include <map>
#include <iostream>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

using namespace std;


class EcalLaserDbAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalLaserDbAnalyzer( const edm::ParameterSet& );
  ~EcalLaserDbAnalyzer ();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:

  // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalLaserDbAnalyzer::EcalLaserDbAnalyzer( const edm::ParameterSet& iConfig )
{

}


EcalLaserDbAnalyzer::~EcalLaserDbAnalyzer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalLaserDbAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  // get record from offline DB
  edm::ESHandle<EcalLaserDbService> pSetup;
  iSetup.get<EcalLaserDbRecord>().get( pSetup );
  std::cout << "EcalLaserDbAnalyzer::analyze-> got EcalLaserDbRecord: " << std::endl;


  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  const EcalLaserAPDPNRatios* myapdpn =  pSetup->getAPDPNRatios();
  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap =  myapdpn->getLaserMap();

//   EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
//   const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap =  myapdpn->getTimeMap();

  EcalLaserAPDPNref apdpnref;
  const EcalLaserAPDPNRatiosRef* myapdpnref =  pSetup->getAPDPNRatiosRef();
  const EcalLaserAPDPNRatiosRefMap& laserRefMap =  myapdpnref->getMap();

  EcalLaserAlpha alpha;
  const EcalLaserAlphas* myalpha =  pSetup->getAlphas();
  const EcalLaserAlphaMap& laserAlphaMap =  myalpha->getMap();

  //  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMapIterator laserRatiosIter; // Laser iterator
  //  EcalLaserAPDPNRatios::EcalLaserTimeStampMapIterator laserTimeIter;
  //  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMapIterator laserRefIter; 
  //  EcalLaserAlphas::EcalLaserAlphaMapIterator laserAlphaIter; 	  
  
  //   int ieta = 83;
  //   int iphi = 168;
  //   EBDetId ebdetid(ieta,iphi);
  
  //   // use a channel to fetch values from DB
  //   double r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
  //   int ieta =  int( 1 + r1*85 );
  //   r1 = (double)std::rand()/( double(RAND_MAX)+double(1) );
  //   int iphi =  int( 1 + r1*20 );
  //   EBDetId ebdetid(ieta,iphi); //eta,phi
  //   std::cout << "*** XTAL: " << ebdetid << std::endl;
  

   for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
     if(ieta==0) continue;
     for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
       try
 	{
 	  EBDetId ebdetid(ieta,iphi);
          int hi = ebdetid.hashedIndex();

	  // 	  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
	  // 	  iSetup.get< EcalMappingRcd >().get(ecalmapping);
	  // 	  const EcalElectronicsMapping* TheMapping = ecalmapping.product();
	  //  	  int dccid = TheMapping-> DCCid(ebdetid);
	  //  	  int tccid = TheMapping-> TCCid(ebdetid);
	  
 	  std::cout << ebdetid << " " 
 		    << ebdetid.ietaSM() << " " << ebdetid.iphiSM() << " "
		    << hi << std::endl;
	  //<< ebdetid.rawId() << " " 
	  //<< dccid << " " << tccid 
	  //<< std::endl;
	  
	  if (hi< (int)laserRatiosMap.size()) {
	    apdpnpair = laserRatiosMap[hi];
	    std::cout << " APDPN pair " 
		      << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;
	  } else {
	    edm::LogError("EcalLaserDbAnalyzer") << "error with laserRatiosMap!" << endl;     
	  }
	  
// 	  if (iLM-1< (int)laserTimeMap.size()) {
// 	    timestamp = laserTimeMap[iLM-1];  
// 	    std::cout << " TIME pair " 
// 		      << timestamp.t1.value() << " , " << timestamp.t2.value() << std::endl;
// 	  } else {
// 	    edm::LogError("EcalLaserDbAnalyzer") << "error with laserTimeMap!" << endl;     
// 	  }
	  
	  if (hi< (int)laserRefMap.size()) {
	    apdpnref = laserRefMap[hi];
	    std::cout << " APDPN ref " << apdpnref << std::endl;
	  } else { 
	    edm::LogError("EcalLaserDbAnalyzer") << "error with laserRefMap!" << endl;     
	  }
	  
	  if (hi< (int)laserAlphaMap.size()) {
	    alpha = laserAlphaMap[hi];
	    std::cout << " ALPHA " << alpha << std::endl;
	  } else {
	    edm::LogError("EcalLaserDbAnalyzer") << "error with laserAlphaMap!" << endl;     
	  }
	  
 	}
       catch (...)
	 {
	   std::cout << "Error" << std::endl;
	 }
     }
   }
 


  
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserDbAnalyzer);
