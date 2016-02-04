
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTTtrigPrint.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

namespace edmtest {

  DTTtrigPrint::DTTtrigPrint(edm::ParameterSet const& p) {
  }

  DTTtrigPrint::DTTtrigPrint(int i) {
  }

  DTTtrigPrint::~DTTtrigPrint() {
  }

  void DTTtrigPrint::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTTtrig> tTrig;
    context.get<DTTtrigRcd>().get(tTrig);
    std::cout << tTrig->version() << std::endl;
    std::cout << std::distance( tTrig->begin(), tTrig->end() ) << " data in the container" << std::endl;
    DTTtrig::const_iterator iter = tTrig->begin();
    DTTtrig::const_iterator iend = tTrig->end();
    while ( iter != iend ) {
      const DTTtrigId&   trigId   = iter->first;
      const DTTtrigData& trigData = iter->second;
      float trigTime;
      float trigTrms;
      float trigKfac;
      float trigComp;
      tTrig->get( trigId.wheelId,
                  trigId.stationId,
                  trigId.sectorId,
                  trigId.slId,
                  trigId.layerId,
                  trigId.cellId,
                  trigTime, trigTrms, trigKfac, DTTimeUnits::counts );
      tTrig->get( trigId.wheelId,
                  trigId.stationId,
                  trigId.sectorId,
                  trigId.slId,
                  trigId.layerId,
                  trigId.cellId,
                  trigComp, DTTimeUnits::counts );
      std::cout << trigId.wheelId   << " "
                << trigId.stationId << " "
                << trigId.sectorId  << " "
                << trigId.slId      << " "
                << trigId.layerId   << " "
                << trigId.cellId    << " -> "
                << trigData.tTrig    << " "
                << trigData.tTrms    << " "
                << trigData.kFact    << " -> "
                << trigTime          << " "
                << trigTrms          << " "
                << trigKfac          << " -> "
                << trigComp          << std::endl;
      iter++;
    }
  }
  DEFINE_FWK_MODULE(DTTtrigPrint);
}
