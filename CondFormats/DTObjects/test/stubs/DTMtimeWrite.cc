
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

#include "CondFormats/DTObjects/test/stubs/DTMtimeWrite.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace edmtest {

  DTMtimeWrite::DTMtimeWrite(edm::ParameterSet const& p) {
  }

  DTMtimeWrite::DTMtimeWrite(int i) {
  }

  DTMtimeWrite::~DTMtimeWrite() {
  }

  void DTMtimeWrite::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

  }

  void DTMtimeWrite::endJob() {

    std::cout<<"DTMtimeWrite::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTMtime* mTime = new DTMtime( "cmssw_Mtime" );

    int status = 0;
    std::ifstream ifile( "testMtime.txt" );
    int whe;
    int sta;
    int sec;
    int qua;
    float mti;
    float rms;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> mti
                  >> rms ) {
      status = mTime->set( whe, sta, sec, qua, mti, rms,
                           DTVelocityUnits::cm_per_ns );
//                           DTVelocityUnits::cm_per_count );
//                           DTTimeUnits::ns );
//                           DTTimeUnits::counts );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << mti << " "
                << rms << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }

    chkData( mTime );

    if( dbservice->isNewTagRequest("DTMtimeRcd") ){
      dbservice->createNewIOV<DTMtime>(
                 mTime,dbservice->beginOfTime(),
                       dbservice->endOfTime(),"DTMtimeRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendSinceTime<DTMtime>(
//                 mTime,dbservice->currentTime(),"DTMtimeRcd");
    }

  }

  void DTMtimeWrite::chkData( DTMtime* mTime ) {
    std::ifstream ifile( "testMtime.txt" );
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    float mti;
    float rms;
    float ckmti;
    float ckrms;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> mti
                  >> rms ) {
      status = mTime->get( whe, sta, sec, qua, ckmti, ckrms,
                           DTTimeUnits::counts );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << mti << " "
                << rms << "  -> "
                << ckmti << " "
                << ckrms << "  -> ";
      std::cout << "get status: " << status << std::endl;
    }
    return;
  }

  DEFINE_FWK_MODULE(DTMtimeWrite);
}
