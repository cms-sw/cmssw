
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DTObjects/test/stubs/DTTtrigWrite.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace edmtest {

  DTTtrigWrite::DTTtrigWrite(edm::ParameterSet const& p) {
  }

  DTTtrigWrite::DTTtrigWrite(int i) {
  }

  DTTtrigWrite::~DTTtrigWrite() {
  }

  void DTTtrigWrite::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

    std::cout<<"DTTtrigWrite::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTTtrig* tTrig = new DTTtrig( "cmssw_tTrig" );

    int status = 0;
    std::ifstream ifile( "testTtrig.txt" );
    int whe;
    int sta;
    int sec;
    int qua;
    float tri;
    float rms;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> tri
                  >> rms ) {
      status = tTrig->setSLTtrig( whe, sta, sec, qua, tri, rms );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << tri << " "
                << rms << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }

    if( dbservice->isNewTagRequest("DTTtrigRcd") ){
      dbservice->createNewIOV<DTTtrig>(
                 tTrig,dbservice->endOfTime(),"DTTtrigRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendSinceTime<DTTtrig>(
//                 tTrig,dbservice->currentTime(),"DTTtrigRcd");
    }

  }
  DEFINE_FWK_MODULE(DTTtrigWrite);
}
