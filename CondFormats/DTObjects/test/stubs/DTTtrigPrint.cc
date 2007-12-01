
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
  }
  DEFINE_FWK_MODULE(DTTtrigPrint);
}
