
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

#include "CondFormats/DTObjects/test/stubs/DTMtimePrint.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"

namespace edmtest {

  DTMtimePrint::DTMtimePrint(edm::ParameterSet const& p) {
  }

  DTMtimePrint::DTMtimePrint(int i) {
  }

  DTMtimePrint::~DTMtimePrint() {
  }

  void DTMtimePrint::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<DTMtime> mTime;
    context.get<DTMtimeRcd>().get(mTime);
    std::cout << mTime->version() << std::endl;
    std::cout << std::distance( mTime->begin(), mTime->end() ) << " data in the container" << std::endl;
  }
  DEFINE_FWK_MODULE(DTMtimePrint);
}
