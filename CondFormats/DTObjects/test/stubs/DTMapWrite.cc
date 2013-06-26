
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

#include "CondFormats/DTObjects/test/stubs/DTMapWrite.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace edmtest {

  DTMapWrite::DTMapWrite(edm::ParameterSet const& p) {
  }

  DTMapWrite::DTMapWrite(int i) {
  }

  DTMapWrite::~DTMapWrite() {
  }

  void DTMapWrite::analyze(const edm::Event& e,
                           const edm::EventSetup& context) {

    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

  }

  void DTMapWrite::endJob() {

    std::cout<<"DTMapWrite::analyze "<<std::endl;
    edm::Service<cond::service::PoolDBOutputService> dbservice;
    if( !dbservice.isAvailable() ){
      std::cout<<"db service unavailable"<<std::endl;
      return;
    }

    DTReadOutMapping* ro_map = new DTReadOutMapping( "cmssw_ROB",
                                                     "cmssw_ROS" );
    int status = 0;
    std::ifstream ifile( "testMap.txt" );
    int ddu;
    int ros;
    int rob;
    int tdc;
    int cha;
    int whe;
    int sta;
    int sec;
    int qua;
    int lay;
    int cel;
    while ( ifile >> ddu
                  >> ros
                  >> rob
                  >> tdc
                  >> cha
                  >> whe
                  >> sta
                  >> sec
                  >> qua
                  >> lay
                  >> cel ) {
      status = ro_map->insertReadOutGeometryLink( ddu, ros, rob, tdc, cha,
                                                  whe, sta, sec,
                                                  qua, lay, cel );
      std::cout << ddu << " "
                << ros << " "
                << rob << " "
                << tdc << " "
                << cha << " "
                << whe << " "
                << sta << " "
                << sec << " "
                << qua << " "
                << lay << " "
                << cel << "  -> ";                
      std::cout << "insert status: " << status << std::endl;
    }
    if( dbservice->isNewTagRequest("DTReadOutMappingRcd") ){
      dbservice->createNewIOV<DTReadOutMapping>(
                 ro_map,dbservice->beginOfTime(),
                        dbservice->endOfTime(),"DTReadOutMappingRcd");
    }
    else{
      std::cout << "already present tag" << std::endl;
//      dbservice->appendSinceTime<DTReadOutMapping>(
//                 ro_map,dbservice->currentTime(),"DTReadOutMappingRcd");
    }
  }
  DEFINE_FWK_MODULE(DTMapWrite);
}
