
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DTObjects/test/stubs/DTDeadWrite.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

namespace edmtest {

  DTDeadWrite::DTDeadWrite(edm::ParameterSet const& p) {
  }

  DTDeadWrite::DTDeadWrite(int i) {
  }

  DTDeadWrite::~DTDeadWrite() {
  }

  void DTDeadWrite::analyze(const edm::Event& e,
                           const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

  }

  void DTDeadWrite::endJob() {

    std::cout<<"DTDeadWrite::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTDeadFlag* dlist = new DTDeadFlag( "deadList" );

    fill_dead_HV( "dead_HV_list.txt", dlist );
    fill_dead_TP( "dead_TP_list.txt", dlist );
    fill_dead_RO( "dead_RO_list.txt", dlist );
    fill_discCat( "discCat_list.txt", dlist );

    if( dbservice->isNewTagRequest("DTDeadFlagRcd") ){
      dbservice->createNewIOV<DTDeadFlag>(
                 dlist,dbservice->beginOfTime(),
                       dbservice->endOfTime(),"DTDeadFlagRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
      int currentRun = 10;
//      dbservice->appendTillTime<DTDeadFlag>(
      dbservice->appendSinceTime<DTDeadFlag>(
                 dlist,currentRun,"DTDeadFlagRcd");
//      dbservice->appendSinceTime<DTDeadFlag>(
//                 dlist,dbservice->currentTime(),"DTDeadFlagRcd");
    }
  }

  void DTDeadWrite::fill_dead_HV( const char* file, DTDeadFlag* deadList ) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile( file );
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel ) {
      status = deadList->setCellDead_HV( whe, sta, sec, qua, lay, cel, true );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }
  void DTDeadWrite::fill_dead_TP( const char* file, DTDeadFlag* deadList ) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile( file );
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel ) {
      status = deadList->setCellDead_TP( whe, sta, sec, qua, lay, cel, true );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }
  void DTDeadWrite::fill_dead_RO( const char* file, DTDeadFlag* deadList ) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile( file );
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel ) {
      status = deadList->setCellDead_RO( whe, sta, sec, qua, lay, cel, true );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }
  void DTDeadWrite::fill_discCat( const char* file, DTDeadFlag* deadList ) {
    int status = 0;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    std::ifstream ifile( file );
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel ) {
      status = deadList->setCellDiscCat( whe, sta, sec, qua, lay, cel, true );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }
    return;
  }

  DEFINE_FWK_MODULE(DTDeadWrite);
}
