
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

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

#include "CondFormats/DTObjects/test/stubs/DTT0Print.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

using namespace std;

namespace edmtest {

  DTT0Print::DTT0Print(edm::ParameterSet const& p) {
  }

  DTT0Print::DTT0Print(int i) {
  }

  DTT0Print::~DTT0Print() {
  }

  void DTT0Print::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTT0> t0;
    context.get<DTT0Rcd>().get(t0);
    std::cout << t0->version() << std::endl;
    std::cout << std::distance( t0->begin(), t0->end() ) << " data in the container" << std::endl;
  }
  DEFINE_FWK_MODULE(DTT0Print);
}
