
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DTObjects/test/stubs/DTConfigWrite.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace edmtest {

  DTConfigWrite::DTConfigWrite(edm::ParameterSet const& p) {
  }

  DTConfigWrite::DTConfigWrite(int i) {
  }

  DTConfigWrite::~DTConfigWrite() {
  }

  void DTConfigWrite::analyze(const edm::Event& e,
                              const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

  }

  void DTConfigWrite::endJob() {

    std::cout<<"DTConfigWrite::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTCCBConfig* conf = new DTCCBConfig( "test_config" );

    int status = 0;
    std::ifstream ifile( "testConfig.txt" );
    int run;
    int nty;
    int kty;
    int key;
    int whe;
    int sta;
    int sec;
    int nbr;
    int ibr;
    ifile >> run >> nty;
    conf->setStamp( run );
    std::vector<DTConfigKey> fullKey;
    while ( nty-- ) {
      ifile >> kty >> key;
      DTConfigKey confList;
      confList.confType = kty;
      confList.confKey  = key;
      fullKey.push_back( confList );
    }
    conf->setFullKey( fullKey );
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> nbr ) {
      std::vector<int> cfg;
      while ( nbr-- ) {
        ifile >> ibr;
        cfg.push_back( ibr );
      }
      status = conf->setConfigKey( whe, sta, sec, cfg );
      std::cout << whe << " "
                << sta << " "
                << sec << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }

    std::cout << "end of time : " << dbservice->endOfTime() << std::endl;
    if( dbservice->isNewTagRequest("DTCCBConfigRcd") ){
      dbservice->createNewIOV<DTCCBConfig>(
                 conf,dbservice->beginOfTime(),
                      dbservice->endOfTime(),"DTCCBConfigRcd");
//                      0xffffffff,"DTCCBConfigRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendSinceTime<DTCCBConfig>(
//                 conf,dbservice->currentTime(),"DTCCBConfigRcd");
    }

  }
  DEFINE_FWK_MODULE(DTConfigWrite);
}
