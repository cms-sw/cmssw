/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"

using namespace std;
namespace edmtest {
  class CSCReadCrateMapValuesAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCReadCrateMapValuesAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCReadCrateMapValuesAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    const edm::ESGetToken<CSCCrateMap, CSCCrateMapRcd> token_;
  };

  void CSCReadCrateMapValuesAnalyzer::analyze(edm::StreamID,
                                              const edm::Event& e,
                                              const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("CSCCrateMap");

    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCCrateMap* myCrateMap = &context.getData(token_);

    std::map<int, CSCMapItem::MapItem>::const_iterator it;

    int count = 0;
    for (it = myCrateMap->crate_map.begin(); it != myCrateMap->crate_map.end(); ++it) {
      count = count + 1;
      log << "Key: crate " << it->first << std::endl;

      log << count << ") ";
      log << it->second.chamberLabel << "  ";
      log << it->second.chamberId << "  ";
      log << it->second.endcap << "  ";
      log << it->second.station << "  ";
      log << it->second.ring << "  ";
      log << it->second.chamber << "  ";
      log << it->second.cscIndex << "  ";
      log << it->second.layerIndex << "  ";
      log << it->second.stripIndex << "  ";
      log << it->second.anodeIndex << "  ";
      log << it->second.strips << "  ";
      log << it->second.anodes << "  ";
      log << it->second.crateLabel << "  ";
      log << it->second.crateid << "  ";
      log << it->second.sector << "  ";
      log << it->second.trig_sector << "  ";
      log << it->second.dmb << "  ";
      log << it->second.cscid << "  ";
      log << it->second.ddu << "  ";
      log << it->second.ddu_input << "  ";
      log << it->second.slink << "  "
          << "  ";
      log << it->second.fed_crate << "  "
          << "  ";
      log << it->second.ddu_slot << "  "
          << "  ";
      log << it->second.dcc_fifo << "  "
          << "  ";
      log << it->second.fiber_crate << "  "
          << "  ";
      log << it->second.fiber_pos << "  "
          << "  ";
      log << it->second.fiber_socket << "  " << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadCrateMapValuesAnalyzer);
}  // namespace edmtest
