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

#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"


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

        for(int ieta=-EBDetId::MAX_IETA; ieta<=EBDetId::MAX_IETA; ++ieta) {
                if(ieta==0) continue;
                for(int iphi=EBDetId::MIN_IPHI; iphi<=EBDetId::MAX_IPHI; ++iphi) {
                        EBDetId ebdetid(ieta,iphi);

                        std::cout << ebdetid << " " << ebdetid.ietaSM() << " " << ebdetid.iphiSM() << std::endl;

                        EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator itratio = laserRatiosMap.find( ebdetid );
                        if (itratio != laserRatiosMap.end()) {
                                apdpnpair = (*itratio);
                                std::cout << " APDPN pair = " << apdpnpair.p1 << " , " << apdpnpair.p2 << std::endl;
                        } else {
                                edm::LogError("EcalLaserDbService") << "error with laserRatiosMap!" << endl;
                        }

                        EcalLaserAPDPNRatiosRefMap::const_iterator itref = laserRefMap.find( ebdetid );
                        if ( itref != laserRefMap.end() ) {
                                apdpnref = (*itref);
                                std::cout << " APDPN ref = " << apdpnref << std::endl;
                        } else {
                                edm::LogError("EcalLaserDbService") << "error with laserRefMap!" << endl;
                        }

                        EcalLaserAlphaMap::const_iterator italpha = laserAlphaMap.find( ebdetid );
                        if ( italpha != laserAlphaMap.end() ) {
                                alpha = (*italpha);
                                std::cout << " ALPHA = " << alpha << std::endl;
                        } else {
                                edm::LogError("EcalLaserDbService") << "error with laserAlphaMap!" << endl;
                        }
                }
        }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserDbAnalyzer);
