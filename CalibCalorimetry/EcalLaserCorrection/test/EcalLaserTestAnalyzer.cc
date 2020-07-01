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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

using namespace std;
//using namespace oracle::occi;

class EcalLaserTestAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalLaserTestAnalyzer(const edm::ParameterSet&);
  ~EcalLaserTestAnalyzer() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

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
EcalLaserTestAnalyzer::EcalLaserTestAnalyzer(const edm::ParameterSet& iConfig)
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

EcalLaserTestAnalyzer::~EcalLaserTestAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EcalLaserTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //   using namespace edm;

  // get record from offline DB
  edm::ESHandle<EcalLaserDbService> pSetup;
  iSetup.get<EcalLaserDbRecord>().get(pSetup);
  std::cout << "EcalLaserTestAnalyzer::analyze-> got EcalLaserDbRecord: " << std::endl;
  //  pSetup->setVerbosity(true);

  //   int ieta = 83;
  //   int iphi = 168;
  //   EBDetId testid(ieta,iphi);
  //   edm::Timestamp testtime(2222);

  //   edm::ESHandle< EcalElectronicsMapping > ecalmapping;
  //   iSetup.get< EcalMappingRcd >().get(ecalmapping);
  //   const EcalElectronicsMapping* TheMapping = ecalmapping.product();
  // //   int dccid = TheMapping-> DCCid(testid);
  // //   int tccid = TheMapping-> TCCid(testid);

  // //   std::cout << std::endl
  // // 	    << "TESTID: " << testid << " "
  // // 	    << testid.ietaSM() << " " << testid.iphiSM() << " "
  // // 	    << testid.rawId() << " "
  // // 	    << dccid << " " << tccid
  // // 	    << std::endl;

  //   float blah = pSetup->getLaserCorrection(testid, testtime);
  //   std::cout << " EcalLaserTestAnalyzer: " << blah << std::endl;

  std::cout << "---> FIRST ECAL BARREL " << endl;

  // ECAL Barrel
  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      try {
        EBDetId testid(ieta, iphi);
        //	  edm::Timestamp testtime(12000);

        // 	  int dccid = TheMapping-> DCCid(testid);
        // 	  int tccid = TheMapping-> TCCid(testid);

        // 	  EcalElectronicsId myid = TheMapping->getElectronicsId(testid);
        // 	  EcalTriggerElectronicsId mytid = TheMapping->getTriggerElectronicsId(testid);

        std::cout << std::endl
                  << "CRYSTAL EB: " << testid
                  << " "
                  // 		    << testid.ietaSM() << " " << testid.iphiSM() << " : "
                  // 		    << testid.rawId() << " : " << myid << " " << myid.rawId() << " : "
                  // 	  	    << dccid << " " << tccid
                  << std::endl;
        //	  std::cout << testid << std::endl;

        float blah = pSetup->getLaserCorrection(testid, iEvent.time());
        std::cout << " EcalLaserTestAnalyzer: " << iEvent.time().value() << " " << blah << std::endl;

      } catch (...) {
      }
    }
  }

  std::cout << "---> NOW ECAL ENDCAP " << endl;

  //   ECAL Endcap
  for (int iX = EEDetId::IX_MIN; iX <= EEDetId::IX_MAX; ++iX) {
    for (int iY = EEDetId::IY_MIN; iY <= EEDetId::IY_MAX; ++iY) {
      // make an EEDetId since we need EEDetId::rawId() to be used as the key for the pedestals
      try {
        // +Z side
        EEDetId testidpos(iX, iY, 1);
        // 	  edm::Timestamp testtime(12000);
        //	  std::cout << " EcalLaserTestAnalyzer: " << testidpos << " " << testidpos.isc() << endl;

        // 	  // test of elec mapping
        // 	  EcalElectronicsId myidpos = TheMapping->getElectronicsId(testidpos);
        std::cout << std::endl
                  << "CRYSTAL EE+: " << testidpos << " " << testidpos.isc()
                  << " "
                  // 		    << testidpos.rawId() << " : " << myidpos << " " << myidpos.rawId() << " : "
                  // 		    << myidpos.dccId()
                  << std::endl;
        // 	  //

        float blah = pSetup->getLaserCorrection(testidpos, iEvent.time());
        std::cout << " EcalLaserTestAnalyzer: " << iEvent.time().value() << " " << blah << std::endl;

        // -Z side
        EEDetId testidneg(iX, iY, -1);
        //	  std::cout << " EcalLaserTestAnalyzer: " << testidneg << " " << testidneg.isc() << endl;

        // 	  EcalElectronicsId myidneg = TheMapping->getElectronicsId(testidneg);
        std::cout << std::endl
                  << "CRYSTAL EE-: " << testidneg << " " << testidneg.isc()
                  << " "
                  // 		    << testidneg.rawId() << " : " << myidneg << " " << myidneg.rawId() << " : "
                  // 		    << myidneg.dccId()
                  << std::endl;

        blah = pSetup->getLaserCorrection(testidneg, iEvent.time());
        std::cout << " EcalLaserTestAnalyzer: " << iEvent.time().value() << " " << blah << std::endl;
      } catch (...) {
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserTestAnalyzer);
