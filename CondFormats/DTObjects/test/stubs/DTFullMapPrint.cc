
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.
It gets a compact map from the EventSetup and dumps it as a full map.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTFullMapPrint.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

namespace edmtest {

  DTFullMapPrint::DTFullMapPrint(edm::ParameterSet const& p) {
  }

  DTFullMapPrint::DTFullMapPrint(int i) {
  }

  DTFullMapPrint::~DTFullMapPrint() {
  }

  void DTFullMapPrint::analyze(const edm::Event& e,
                                  const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTReadOutMapping> dbMap;
    context.get<DTReadOutMappingRcd>().get(dbMap);
    const DTReadOutMapping* ro_map = dbMap->fullMap();
    std::cout << ro_map->mapCellTdc() << " - "
              << ro_map->mapRobRos()  << std::endl;
    std::cout << std::distance( ro_map->begin(), ro_map->end() ) << " connections in the map" << std::endl;
    DTReadOutMapping::const_iterator iter = ro_map->begin();
    DTReadOutMapping::const_iterator iend = ro_map->end();
    while ( iter != iend ) {
      const DTReadOutGeometryLink& link = *iter++;
      std::cout << link.dduId << " "
                << link.rosId << " "
                << link.robId << " "
                << link.tdcId << " "
                << link.channelId << " "
                << link.wheelId << " "
                << link.stationId << " "
                << link.sectorId << " "
                << link.slId << " "
                << link.layerId << " "
                << link.cellId << std::endl;
    }
  }
  DEFINE_FWK_MODULE(DTFullMapPrint);
}
