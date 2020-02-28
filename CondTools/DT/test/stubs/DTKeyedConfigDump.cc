
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondTools/DT/test/stubs/DTKeyedConfigDump.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"

namespace edmtest {

  DTKeyedConfigDump::DTKeyedConfigDump(edm::ParameterSet const& p)
      : dumpCCBKeys{p.getParameter<bool>("dumpCCBKeys")},
        dumpAllData{p.getParameter<bool>("dumpAllData")},
        configToken_{esConsumes<DTCCBConfig, DTCCBConfigRcd>()} {
    if (dumpCCBKeys) {
      keyListToken_ = esConsumes<cond::persistency::KeyList, DTKeyedConfigListRcd>();
    }
  }

  void DTKeyedConfigDump::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;

    // get configuration for current run
    auto const& conf = context.getData(configToken_);

    std::cout << conf.version() << std::endl;
    std::cout << std::distance(conf.begin(), conf.end()) << " data in the container" << std::endl;
    edm::ValidityInterval iov(context.get<DTCCBConfigRcd>().validityInterval());
    unsigned int currValidityStart = iov.first().eventID().run();
    unsigned int currValidityEnd = iov.last().eventID().run();
    std::cout << "valid since run " << currValidityStart << " to run " << currValidityEnd << std::endl;

    if (!dumpCCBKeys)
      return;

    const DTKeyedConfig** allBricks = new const DTKeyedConfig*[100000];
    int nBricks = 0;

    auto const& keyList = context.getData(keyListToken_);
    // loop over chambers
    DTCCBConfig::ccb_config_map configKeys(conf.configKeyMap());
    DTCCBConfig::ccb_config_iterator iter = configKeys.begin();
    DTCCBConfig::ccb_config_iterator iend = configKeys.end();
    while (iter != iend) {
      // get chamber id
      const DTCCBId& ccbId = iter->first;
      std::cout << ccbId.wheelId << " " << ccbId.stationId << " " << ccbId.sectorId << " -> ";
      std::cout << std::endl;
      // get brick identifiers list
      const std::vector<int>& ccbConf = iter->second;
      std::vector<int>::const_iterator cfgIter = ccbConf.begin();
      std::vector<int>::const_iterator cfgIend = ccbConf.end();

      // loop over configuration bricks
      while (cfgIter != cfgIend) {
        // get brick identifier
        int id = *cfgIter++;
        std::cout << " " << id;
        std::cout << std::endl;
        if (!dumpAllData)
          continue;
        const DTKeyedConfig* kBrick = 0;
        cfgCache.get(keyList, id, kBrick);
        allBricks[nBricks++] = kBrick;
        if (kBrick == 0) {
          std::cout << "brick missing" << std::endl;
          continue;
        }
        std::vector<std::string>::const_iterator s_iter = kBrick->dataBegin();
        std::vector<std::string>::const_iterator s_iend = kBrick->dataEnd();
        while (s_iter != s_iend)
          std::cout << "        ----> " << *s_iter++ << std::endl;
      }
      std::cout << std::endl;
      ++iter;
    }
  }
  DEFINE_FWK_MODULE(DTKeyedConfigDump);
}  // namespace edmtest
