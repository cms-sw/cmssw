
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/DTObjects/test/stubs/DTDeadWrite.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

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

    DTDeadFlag* dlist = new DTDeadFlag( "list1" );
    int status = 0;
    std::ifstream ifile( "deadList.txt" );
    std::cout << " file open" << std::endl;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    while ( ifile >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel ) {
      status = dlist->setCellDead( whe, sta, sec, qua, lay, cel, true );
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }
    if( dbservice->isNewTagRequest("DTDeadFlagRcd") ){
      dbservice->createNewIOV<DTDeadFlag>(
                 dlist,dbservice->endOfTime(),"DTDeadFlagRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendTillTime<DTDeadFlag>(
      dbservice->appendSinceTime<DTDeadFlag>(
                 dlist,10,"DTDeadFlagRcd");
//      dbservice->appendSinceTime<DTDeadFlag>(
//                 dlist,dbservice->currentTime(),"DTDeadFlagRcd");
    }
  }
  DEFINE_FWK_MODULE(DTDeadWrite);
}
