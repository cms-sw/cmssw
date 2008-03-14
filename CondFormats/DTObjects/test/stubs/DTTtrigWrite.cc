
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

#include "CondFormats/DTObjects/test/stubs/DTTtrigWrite.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"

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

  }

  void DTTtrigWrite::endJob() {

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
      status = tTrig->set( whe, sta, sec, qua, tri, rms );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << tri << " "
                << rms << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }

    chkData( tTrig );

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

  void DTTtrigWrite::chkData( DTTtrig* tTrig ) {
    std::ifstream ifile( "testTtrig.txt" );
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    float tri;
    float rms;
    float cktri;
    float ckrms;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> tri
                  >> rms ) {
      status = tTrig->get( whe, sta, sec, qua, cktri, ckrms );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << tri << " "
                << rms << "  -> "
                << cktri << " "
                << ckrms << "  -> ";
      std::cout << "get status: " << status << std::endl;
    }
    return;
  }

  DEFINE_FWK_MODULE(DTTtrigWrite);
}
