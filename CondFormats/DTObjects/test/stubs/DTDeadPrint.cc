
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTDeadPrint.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

namespace edmtest {

  DTDeadPrint::DTDeadPrint(edm::ParameterSet const& p) {
  }

  DTDeadPrint::DTDeadPrint(int i) {
  }

  DTDeadPrint::~DTDeadPrint() {
  }

  void DTDeadPrint::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
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
                << st.dead_HV  << " "
                << st.dead_TP  << " "
                << st.dead_RO  << " "
                << st.discCat  << std::endl;
    }
  }
  DEFINE_FWK_MODULE(DTDeadPrint);
}
