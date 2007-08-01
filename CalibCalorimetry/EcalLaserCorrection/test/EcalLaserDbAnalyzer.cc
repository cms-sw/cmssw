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
//using namespace oracle::occi;


class EcalLaserDbAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalLaserDbAnalyzer( const edm::ParameterSet& );
  ~EcalLaserDbAnalyzer ();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  //  std::string m_timetype;
  //  std::map<std::string, unsigned long long> m_cacheIDs;
  //  std::map<std::string, std::string> m_records;
  //  unsigned long m_firstRun ;
  //  unsigned long m_lastRun ;

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
  //:
  //  m_timetype(iConfig.getParameter<std::string>("timetype")),
  //  m_cacheIDs(),
  //  m_records()
{
  //   std::cout << "EcalLaserDbAnalyzer::EcalLaserDbAnalyzer->... construct me!" << std::endl;
   //now do what ever initialization is needed

//    std::string container;
//    std::string tag;
//    std::string record;
  
//    //   m_firstRun=(unsigned long)atoi( iConfig.getParameter<std::string>("firstRun").c_str());
//    //   m_lastRun=(unsigned long)atoi( iConfig.getParameter<std::string>("lastRun").c_str());
  
//    typedef std::vector< edm::ParameterSet > Parameters;
//    Parameters toGet = iConfig.getParameter<Parameters>("toGet");
//    for(Parameters::iterator i = toGet.begin(); i != toGet.end(); ++i) {
//      container = i->getParameter<std::string>("container");
//      record = i->getParameter<std::string>("record");
//      m_cacheIDs.insert( std::make_pair(container, 0) );
//      m_records.insert( std::make_pair(container, record) );

//    } 

}


EcalLaserDbAnalyzer::~EcalLaserDbAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalLaserDbAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  //   using namespace edm;

  // get record from offline DB
  edm::ESHandle<EcalLaserDbService> pSetup;
  iSetup.get<EcalLaserDbRecord>().get( pSetup );
  std::cout << "EcalLaserDbAnalyzer::analyze-> got EcalLaserDbRecord: " << std::endl;


  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  const EcalLaserAPDPNRatios* myapdpn =  pSetup->getAPDPNRatios();
  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap =  myapdpn->getLaserMap();

  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  //  const EcalLaserAPDPNRatios* myapdpn =  pSetup->getAPDPNRatios();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap =  myapdpn->getTimeMap();

  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNref apdpnref;
  const EcalLaserAPDPNRatiosRef* myapdpnref =  pSetup->getAPDPNRatiosRef();
  const EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMap& laserRefMap =  myapdpnref->getMap();

  EcalLaserAlphas::EcalLaserAlpha alpha;
  const EcalLaserAlphas* myalpha =  pSetup->getAlphas();
  const EcalLaserAlphas::EcalLaserAlphaMap& laserAlphaMap =  myalpha->getMap();

  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMapIterator laserRatiosIter; // Laser iterator
  EcalLaserAPDPNRatios::EcalLaserTimeStampMapIterator laserTimeIter;
  EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRefMapIterator laserRefIter; 
  EcalLaserAlphas::EcalLaserAlphaMapIterator laserAlphaIter; 	  

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

	  edm::ESHandle< EcalElectronicsMapping > ecalmapping;
	  iSetup.get< EcalMappingRcd >().get(ecalmapping);
	  const EcalElectronicsMapping* TheMapping = ecalmapping.product();
 	  int dccid = TheMapping-> DCCid(ebdetid);
 	  int tccid = TheMapping-> TCCid(ebdetid);
	  
	  std::cout << ebdetid << " " 
		    << ebdetid.ietaSM() << " " << ebdetid.iphiSM() << " "
		    << ebdetid.rawId() << " " 
		    << dccid << " " << tccid 
		    << std::endl;



 	  //	  double r = (double)std::rand()/( double(RAND_MAX)+double(1) );
 	  //	  ical->setValue( ebid.rawId(), laserAlphaMean_ + r*laserAlphaSigma_ );
	  
	  laserRatiosIter = laserRatiosMap.find(ebdetid.rawId());
	  if( laserRatiosIter != laserRatiosMap.end() ) {
	    apdpnpair = laserRatiosIter->second;
	    std::cout << " APDPN pair " << ieta << " " << iphi << " " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;
	  } else {
	    std::cout << " ERROR!  you screwed up!" << std::endl;
	  }

	  laserTimeIter = laserTimeMap.find(ebdetid.ism());
	  if( laserTimeIter != laserTimeMap.end() ) {
	    timestamp = laserTimeIter->second;
	    std::cout << " TIME pair " << ieta << " " << iphi << " " << timestamp.t1.value() << " , " << timestamp.t2.value() << std::endl;
	  } else {
	    std::cout << " ERROR!  you screwed up!" << std::endl;
	  }
	  
	  laserRefIter = laserRefMap.find(ebdetid.rawId());
	  if( laserRefIter != laserRefMap.end() ) {
	    apdpnref = laserRefIter->second;
	    std::cout << " APDPN ref " << ieta << " " << iphi << " " << apdpnref << std::endl;
	  } else {
	    std::cout << " ERROR!  you screwed up!" << std::endl;
	  }
	  
	  laserAlphaIter = laserAlphaMap.find(ebdetid.rawId());
	  if( laserAlphaIter != laserAlphaMap.end() ) {
	    alpha = laserAlphaIter->second;    
	    std::cout << " ALPHA " << ieta << " " << iphi << " " << alpha << std::endl;
	    
	  } else {
	    std::cout << " ERROR!  you screwed up!" << std::endl;
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
