
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondTools/DT/test/stubs/DTFullMapDump.h"
#include "CondTools/DT/interface/DTExpandMap.h"

namespace edmtest {

  DTFullMapDump::DTFullMapDump(edm::ParameterSet const& p) {
// parameters to setup 
    fileName   = p.getParameter< std::string >( "fileName" );
  }

  DTFullMapDump::DTFullMapDump(int i) {
  }

  DTFullMapDump::~DTFullMapDump() {
  }

  void DTFullMapDump::analyze( const edm::Event& e,
                               const edm::EventSetup& context ) {
  }

  void DTFullMapDump::endJob() {
    std::ifstream mapFile( fileName.c_str() );
    DTExpandMap::expandSteering( mapFile );
  }

  DEFINE_FWK_MODULE(DTFullMapDump);
}
