
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondFormats/DTObjects/test/stubs/DTConfigPrint.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"

namespace edmtest {

  DTConfigPrint::DTConfigPrint(edm::ParameterSet const& p) {
// parameters to setup 
    connect   = p.getParameter< std::string >("connect");
    auth_path = p.getParameter< std::string >("authenticationPath");
    token     = p.getParameter< std::string >("token");
    local     = p.getParameter< bool        >("siteLocalConfig");
    if ( local ) catalog = "";
    else         catalog = p.getParameter< std::string >("catalog");
  }

  DTConfigPrint::DTConfigPrint(int i) {
  }

  DTConfigPrint::~DTConfigPrint() {
  }

  void DTConfigPrint::analyze(const edm::Event& e,
                          const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;

// get configuration for current run
    edm::ESHandle<DTCCBConfig> conf;
    context.get<DTCCBConfigRcd>().get(conf);
    std::cout << conf->version() << std::endl;
    std::cout << std::distance( conf->begin(), conf->end() ) << " data in the container" << std::endl;

// loop over chambers
    DTCCBConfig::ccb_config_map configKeys( conf->configKeyMap() );
    DTCCBConfig::ccb_config_iterator iter = configKeys.begin();
    DTCCBConfig::ccb_config_iterator iend = configKeys.end();
    while ( iter != iend ) {
// get chamber id
      const DTCCBId& ccbId = iter->first;
      std::cout << "chamber "
                << ccbId.wheelId   << " "
                << ccbId.stationId << " "
                << ccbId.sectorId  << " -> ";
      std::cout << std::endl;
// get brick identifiers list
      const std::vector<int>& ccbConf = iter->second;
      std::vector<int>::const_iterator cfgIter = ccbConf.begin();
      std::vector<int>::const_iterator cfgIend = ccbConf.end();

// loop over configuration bricks
      while ( cfgIter != cfgIend ) {
// get brick identifier
        int id = *cfgIter++;
        std::cout << " " << id;
        std::cout << std::endl;
      }
      std::cout << std::endl;
      ++iter;
    }
  }
  DEFINE_FWK_MODULE(DTConfigPrint);
}
