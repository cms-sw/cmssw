
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTRangeT0Print.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DataRecord/interface/DTRangeT0Rcd.h"

namespace edmtest {

  DTRangeT0Print::DTRangeT0Print(edm::ParameterSet const& p) {
  }

  DTRangeT0Print::DTRangeT0Print(int i) {
  }

  DTRangeT0Print::~DTRangeT0Print() {
  }

  void DTRangeT0Print::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTRangeT0> t0;
    context.get<DTRangeT0Rcd>().get(t0);
    std::cout << t0->version() << std::endl;
    std::cout << std::distance( t0->begin(), t0->end() ) << " data in the container" << std::endl;
  }
  DEFINE_FWK_MODULE(DTRangeT0Print);
}
