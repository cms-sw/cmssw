
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

#include "CondFormats/DTObjects/test/stubs/DTRangeT0Write.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace edmtest {

  DTRangeT0Write::DTRangeT0Write(edm::ParameterSet const& p) {
  }

  DTRangeT0Write::DTRangeT0Write(int i) {
  }

  DTRangeT0Write::~DTRangeT0Write() {
  }

  void DTRangeT0Write::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

  }

  void DTRangeT0Write::endJob() {

    std::cout<<"DTRangeT0Write::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTRangeT0* rt0 = new DTRangeT0( "cmssw_rt0" );

    int status = 0;
    std::ifstream ifile( "testRT0.txt" );
    int whe;
    int sta;
    int sec;
    int qua;
    int t0min;
    int t0max;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> t0min
                  >> t0max ) {
      status = rt0->set( whe, sta, sec, qua, t0min, t0max );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << t0min << " "
                << t0max << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }

    if( dbservice->isNewTagRequest("DTRangeT0Rcd") ){
      dbservice->createNewIOV<DTRangeT0>(
                 rt0,dbservice->beginOfTime(),
                     dbservice->endOfTime(),"DTRangeT0Rcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendSinceTime<DTRangeT0>(
//                 rt0,dbservice->currentTime(),"DTRangeT0Rcd");
    }

  }
  DEFINE_FWK_MODULE(DTRangeT0Write);
}
