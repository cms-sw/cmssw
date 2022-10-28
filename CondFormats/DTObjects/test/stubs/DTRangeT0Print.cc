
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

namespace edmtest {

  DTRangeT0Print::DTRangeT0Print(edm::ParameterSet const& p) : es_token(esConsumes()) {}

  DTRangeT0Print::DTRangeT0Print(int i) {}

  void DTRangeT0Print::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    const auto& t0 = context.getData(es_token);
    std::cout << t0.version() << std::endl;
    std::cout << std::distance(t0.begin(), t0.end()) << " data in the container" << std::endl;
  }
  DEFINE_FWK_MODULE(DTRangeT0Print);
}  // namespace edmtest
