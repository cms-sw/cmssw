/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <string>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"

//#include "OnlineDB/CSCCondDB/interface/CSCReadChamberMapValuesAnalyzer.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberMapValuesAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCReadChamberMapValuesAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCReadChamberMapValuesAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCChamberMap, CSCChamberMapRcd> token_;
  };

  void CSCReadChamberMapValuesAnalyzer::analyze(edm::StreamID,
                                                const edm::Event& e,
                                                const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("CSCChamberMap");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCChamberMap* myChMap = &context.getData(token_);

    std::map<int, CSCMapItem::MapItem>::const_iterator it;

    int count = 0;
    for (it = myChMap->ch_map.begin(); it != myChMap->ch_map.end(); ++it) {
      count = count + 1;
      log << "Key: chamber " << it->first << std::endl;

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
      log << it->second.slink << "  ";
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
  DEFINE_FWK_MODULE(CSCReadChamberMapValuesAnalyzer);
}  // namespace edmtest
