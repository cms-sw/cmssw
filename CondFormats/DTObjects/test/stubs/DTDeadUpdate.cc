
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

#include "CondFormats/DTObjects/test/stubs/DTDeadUpdate.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

namespace edmtest {

  DTDeadUpdate::DTDeadUpdate(edm::ParameterSet const& p): dSum( 0 ) {
  }

  DTDeadUpdate::DTDeadUpdate(int i): dSum( 0 ) {
  }

  DTDeadUpdate::~DTDeadUpdate() {
  }

  void DTDeadUpdate::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    if ( dSum == 0 ) dSum = new DTDeadFlag( "list2" );
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTDeadFlag> dList;
    context.get<DTDeadFlagRcd>().get(dList);
    std::cout << dList->version() << std::endl;
    std::cout << std::distance( dList->begin(), dList->end() ) << " data in the container" << std::endl;
    DTDeadFlag::const_iterator iter = dList->begin();
    DTDeadFlag::const_iterator iend = dList->end();
    while ( iter != iend ) {
      const std::pair<DTDeadFlagId,DTDeadFlagData>& data = *iter++;
      const DTDeadFlagId&   id = data.first;
      const DTDeadFlagData& st = data.second;
      std::cout << id.  wheelId << " "
                << id.stationId << " "
                << id. sectorId << " "
                << id.     slId << " "
                << id.  layerId << " "
                << id.   cellId << " -> "
                << st.deadFlag  << " "
                << st.nohvFlag  << std::endl;
      if ( st.deadFlag ) dSum->setCellDead( id.  wheelId,
                                            id.stationId,
                                            id. sectorId,
                                            id.     slId,
                                            id.  layerId,
                                            id.   cellId, true );
      if ( st.nohvFlag ) dSum->setCellNoHV( id.  wheelId,
                                            id.stationId,
                                            id. sectorId,
                                            id.     slId,
                                            id.  layerId,
                                            id.   cellId, true );

    }
  }
  void DTDeadUpdate::endJob() {

    std::cout<<"DTDeadUpdate::endJob "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    int status = 0;
    std::ifstream ifile( "testList2.txt" );
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
      std::cout << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  ... " << std::endl;
      status = dSum->setCellDead( whe, sta, sec, qua, lay, cel, true );
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
                 dSum,dbservice->endOfTime(),"DTDeadFlagRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendTillTime<DTDeadFlag>(
      dbservice->appendSinceTime<DTDeadFlag>(
                 dSum,10,"DTDeadFlagRcd");
//      dbservice->appendSinceTime<DTDeadFlag>(
//                 dlist,dbservice->currentTime(),"DTDeadFlagRcd");
    }
  }
  DEFINE_FWK_MODULE(DTDeadUpdate);
}
