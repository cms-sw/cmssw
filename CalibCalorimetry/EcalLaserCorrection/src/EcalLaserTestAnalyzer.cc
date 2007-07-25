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
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

using namespace std;
//using namespace oracle::occi;


class EcalLaserTestAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalLaserTestAnalyzer( const edm::ParameterSet& );
  ~EcalLaserTestAnalyzer ();

  
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
EcalLaserTestAnalyzer::EcalLaserTestAnalyzer( const edm::ParameterSet& iConfig )
  //:
  //  m_timetype(iConfig.getParameter<std::string>("timetype")),
  //  m_cacheIDs(),
  //  m_records()
{
  //   std::cout << "EcalLaserTestAnalyzer::EcalLaserTestAnalyzer->... construct me!" << std::endl;
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


EcalLaserTestAnalyzer::~EcalLaserTestAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalLaserTestAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{

  //   using namespace edm;

  // get record from offline DB
  edm::ESHandle<EcalLaserDbService> pSetup;
  iSetup.get<EcalLaserDbRecord>().get( pSetup );
  std::cout << "EcalLaserTestAnalyzer::analyze-> got EcalLaserDbRecord: " << std::endl;
  //  pSetup->setVerbosity(true);

//   int ieta = 83;
//   int iphi = 168;
//   EBDetId testid(ieta,iphi);
//   edm::Timestamp testtime(2222);

//   float blah = pSetup->getLaserCorrection(testid, testtime);
//   std::cout << " EcalLaserTestAnalyzer: " << blah << std::endl; 

  // ECAL Barrel
  for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
    if(ieta==0) continue;
    for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
      try
 	{
	  
	  EBDetId testid(ieta,iphi);
	  edm::Timestamp testtime(2222);
	  
	  float blah = pSetup->getLaserCorrection(testid, testtime);
	  std::cout << " EcalLaserTestAnalyzer: " << blah << std::endl; 
	  
 	}
      catch (...)
	{
	}
    }
  }
  
  
  // ECAL Endcap
  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try 
	{
	  	  
	  EEDetId testidpos(iX,iY,1);
	  edm::Timestamp testtime(2222);
	  std::cout << " EcalLaserTestAnalyzer: " << testidpos << " " << testidpos.isc() << endl;

	  float blah = pSetup->getLaserCorrection(testidpos, testtime);
	  std::cout << " EcalLaserTestAnalyzer: " << blah << std::endl; 
	  
	  EEDetId testidneg(iX,iY,-1);
	  std::cout << " EcalLaserTestAnalyzer: " << testidneg << " " << testidneg.isc() << endl;
	  blah = pSetup->getLaserCorrection(testidneg, testtime);
	  std::cout << " EcalLaserTestAnalyzer: " << blah << std::endl; 

	}
      catch (...)
	{
	}
    }
  }


}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserTestAnalyzer);
