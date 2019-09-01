
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.
It reads a compact map from ascii file and dumps it as a full map.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondTools/DT/test/stubs/DTFullMapDump.h"
//#include "CondFormats/DTObjects/interface/DTCompactMapAbstractHandler.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
//#include "CondTools/DT/interface/DTExpandMap.h"

namespace edmtest {

  DTFullMapDump::DTFullMapDump(edm::ParameterSet const& p) {
    // parameters to setup
    fileName = p.getParameter<std::string>("fileName");
  }

  DTFullMapDump::DTFullMapDump(int i) {}

  DTFullMapDump::~DTFullMapDump() {}

  void DTFullMapDump::analyze(const edm::Event& e, const edm::EventSetup& context) {}

  void DTFullMapDump::endJob() {
    std::ifstream mapFile(fileName.c_str());
    //    DTExpandMap::expandSteering( mapFile );
    DTReadOutMapping* compMap = new DTReadOutMapping("rob", "ros");
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
    while (mapFile >> ddu >> ros >> rob >> tdc >> cha >> whe >> sta >> sec >> qua >> lay >> cel) {
      //      std::cout << ddu << " "
      //                << ros << " "
      //                << rob << " "
      //                << tdc << " "
      //                << cha << " "
      //                << whe << " "
      //                << sta << " "
      //                << sec << " "
      //                << qua << " "
      //                << lay << " "
      //                << cel << std::endl;
      compMap->insertReadOutGeometryLink(ddu, ros, rob, tdc, cha, whe, sta, sec, qua, lay, cel);
    }

    std::cout << "now expand" << std::endl;
    //    std::cout << DTCompactMapAbstractHandler::getInstance() << std::endl;
    //    DTReadOutMapping* fullMap =
    //                      DTCompactMapAbstractHandler::getInstance()->expandMap(
    //                                                   *compMap );
    const DTReadOutMapping* fullMap = compMap->fullMap();
    std::cout << "done" << std::endl;
    DTReadOutMapping::const_iterator iter = fullMap->begin();
    DTReadOutMapping::const_iterator iend = fullMap->end();
    while (iter != iend) {
      const DTReadOutGeometryLink& link = *iter++;
      std::cout << link.dduId << " " << link.rosId << " " << link.robId << " " << link.tdcId << " " << link.channelId
                << " " << link.wheelId << " " << link.stationId << " " << link.sectorId << " " << link.slId << " "
                << link.layerId << " " << link.cellId << std::endl;
    }
  }

  DEFINE_FWK_MODULE(DTFullMapDump);
}  // namespace edmtest
