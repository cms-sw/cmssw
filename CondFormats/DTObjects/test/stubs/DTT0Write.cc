
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

#include "CondFormats/DTObjects/test/stubs/DTT0Write.h"
#include "CondFormats/DTObjects/interface/DTT0.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace edmtest {

  DTT0Write::DTT0Write(edm::ParameterSet const& p) {
  }

  DTT0Write::DTT0Write(int i) {
  }

  DTT0Write::~DTT0Write() {
  }

  void DTT0Write::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

  }

  void DTT0Write::endJob() {

    std::cout<<"DTT0Write::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTT0* t0 = new DTT0( "cmssw_t0" );

    int status = 0;
    std::ifstream ifile( "testT0.txt" );
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    float t0m;
    float rms;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel
                  >> t0m
                  >> rms ) {
      status = t0->set( whe, sta, sec, qua, lay, cel, t0m, rms,
                        DTTimeUnits::counts );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << " "
                << t0m << " "
                << rms << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }

    if( dbservice->isNewTagRequest("DTT0Rcd") ){
      dbservice->createNewIOV<DTT0>(
                 t0,dbservice->beginOfTime(),
                    dbservice->endOfTime(),"DTT0Rcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendSinceTime<DTT0>(
//                 t0,dbservice->currentTime(),"DTT0Rcd");
    }

  }
  DEFINE_FWK_MODULE(DTT0Write);
}
