
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
    DTT0::const_iterator iter = t0->begin();
    DTT0::const_iterator iend = t0->end();
    while ( iter != iend ) {
      const DTT0Id&   t0Id   = iter->first;
      const DTT0Data& t0Data = iter->second;
      float t0Time;
      float t0Trms;
      t0->get( t0Id.wheelId,
               t0Id.stationId,
               t0Id.sectorId,
               t0Id.slId,
               t0Id.layerId,
               t0Id.cellId,
               t0Time, t0Trms );
      std::cout << t0Id.wheelId   << " "
                << t0Id.stationId << " "
                << t0Id.sectorId  << " "
                << t0Id.slId      << " "
                << t0Id.layerId   << " "
                << t0Id.cellId    << " -> "
                << t0Data.t0mean    << " "
                << t0Data.t0rms     << " -> "
                << t0Time           << " "
                << t0Trms           << std::endl;
      iter++;
    }


  }
  DEFINE_FWK_MODULE(DTT0Print);
}
