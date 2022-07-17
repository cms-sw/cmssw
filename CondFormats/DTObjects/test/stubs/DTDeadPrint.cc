
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

namespace edmtest {

  DTDeadPrint::DTDeadPrint(edm::ParameterSet const& p) : es_token(esConsumes()) {}

  DTDeadPrint::DTDeadPrint(int i) {}

  void DTDeadPrint::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    const auto& dList = context.getData(es_token);
    std::cout << dList.version() << std::endl;
    std::cout << std::distance(dList.begin(), dList.end()) << " data in the container" << std::endl;
    DTDeadFlag::const_iterator iter = dList.begin();
    DTDeadFlag::const_iterator iend = dList.end();
    while (iter != iend) {
      const std::pair<DTDeadFlagId, DTDeadFlagData>& data = *iter++;
      const DTDeadFlagId& id = data.first;
      const DTDeadFlagData& st = data.second;
      std::cout << id.wheelId << " " << id.stationId << " " << id.sectorId << " " << id.slId << " " << id.layerId << " "
                << id.cellId << " -> " << st.dead_HV << " " << st.dead_TP << " " << st.dead_RO << " " << st.discCat
                << std::endl;
    }
  }
  DEFINE_FWK_MODULE(DTDeadPrint);
}  // namespace edmtest
