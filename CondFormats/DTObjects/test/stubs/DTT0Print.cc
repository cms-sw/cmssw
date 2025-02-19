
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTT0Print.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

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
    float t0mean;
    float t0rms;
    DTT0::const_iterator iter = t0->begin();
    DTT0::const_iterator iend = t0->end();
    while ( iter != iend ) {
      const DTT0Data& t0Data = *iter++;
      int channelId = t0Data.channelId;
      if ( channelId == 0 ) continue;
      DTWireId id( channelId );
      DTChamberId* cp = &id;
      DTChamberId ch( *cp );
      DTChamberId cc( id.chamberId() );
      std::cout << channelId   << " "
                <<  id.rawId() << " "
                << cp->rawId() << " "
                <<  ch.rawId() << " "
                <<  cc.rawId() << std::endl;
      t0->get( id, t0mean, t0rms, DTTimeUnits::counts );
      std::cout << id.wheel()      << " "
                << id.station()    << " "
                << id.sector()     << " "
                << id.superlayer() << " "
                << id.layer()      << " "
                << id.wire()       << " -> "
                << t0Data.t0mean   << " "
                << t0Data.t0rms    << " -> "
                << t0mean          << " "
                << t0rms           << std::endl;
    }


  }
  DEFINE_FWK_MODULE(DTT0Print);
}
