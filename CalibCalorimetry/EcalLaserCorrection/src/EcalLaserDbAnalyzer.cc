//
// Toyoko Orimoto (Caltech), 10 July 2007
//


// system include files
#include <memory>
#include <time.h>
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

  //  std::cout << "EcalLaserDbAnalyzer::analyze->..." << std::endl;
  edm::ESHandle<EcalLaserDbService> pSetup;
  iSetup.get<EcalLaserDbRecord>().get( pSetup );
  std::cout << "EcalLaserDbAnalyzer::analyze-> got EcalLaserDbRecord: " << std::endl;
  //  std::cout << "EcalLaserDbAnalyzer::analyze-> getting information for EB channel" << std::endl;

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  const EcalLaserAPDPNRatios* myapdpn =  pSetup->getAPDPNRatios();
  const EcalLaserAPDPNRatiosMap& laserMap =  myapdpn->laser_map;

  //  EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  //  const EcalLaserTimeStampMap&   timeMap  =  myapdpn->time_map;
  //  EcalLaserTimeStampMapIterator timeIter; //TiemStamp iterator
  
  EcalLaserAPDPNRatiosMapIterator laserIter; // Laser iterator
	 
  int iEta = 83;
  int iPhi = 168;
  EBDetId ebdetid(iEta,iPhi);

  laserIter = laserMap.find(ebdetid.rawId());
  if( laserIter != laserMap.end() ) {
    apdpnpair= laserIter->second;
  } else {
    std::cout << "ERROR!  you screwed up!" << std::endl;
    //    continue;
  }

  std::cout << "APDPN pair" << iEta << " " << iPhi << " " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;


}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserDbAnalyzer);
