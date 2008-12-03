
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTMapPrint.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

namespace edmtest {

  DTMapPrint::DTMapPrint(edm::ParameterSet const& p) {
  }

  DTMapPrint::DTMapPrint(int i) {
  }

  DTMapPrint::~DTMapPrint() {
  }

  void DTMapPrint::analyze(const edm::Event& e,
                           const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTReadOutMapping> ro_map;
    context.get<DTReadOutMappingRcd>().get(ro_map);
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
  DEFINE_FWK_MODULE(DTMapPrint);
}
