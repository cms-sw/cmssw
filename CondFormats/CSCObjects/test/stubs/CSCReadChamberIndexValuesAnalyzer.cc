/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <string>
#include <iostream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
#include "CondFormats/DataRecord/interface/CSCChamberIndexRcd.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberIndexValuesAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCReadChamberIndexValuesAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCReadChamberIndexValuesAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCChamberIndex, CSCChamberIndexRcd> token_;
  };

  void CSCReadChamberIndexValuesAnalyzer::analyze(edm::StreamID,
                                                  const edm::Event& e,
                                                  const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("CSCChamberIndex");

    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCChamberIndex* myChIndex = &context.getData(token_);

    std::vector<CSCMapItem::MapItem>::const_iterator it;

    int count = 0;
    for (it = myChIndex->ch_index.begin(); it != myChIndex->ch_index.end(); ++it) {
      count++;

      log << count << ") ";
      log << it->chamberLabel << "  ";
      log << it->chamberId << "  ";
      log << it->endcap << "  ";
      log << it->station << "  ";
      log << it->ring << "  ";
      log << it->chamber << "  ";
      log << it->cscIndex << "  ";
      log << it->layerIndex << "  ";
      log << it->stripIndex << "  ";
      log << it->anodeIndex << "  ";
      log << it->strips << "  ";
      log << it->anodes << "  ";
      log << it->crateLabel << "  ";
      log << it->crateid << "  ";
      log << it->sector << "  ";
      log << it->trig_sector << "  ";
      log << it->dmb << "  ";
      log << it->cscid << "  ";
      log << it->ddu << "  ";
      log << it->ddu_input << "  ";
      log << it->slink << "  ";
      log << it->fed_crate << "  "
          << "  ";
      log << it->ddu_slot << "  "
          << "  ";
      log << it->dcc_fifo << "  "
          << "  ";
      log << it->fiber_crate << "  "
          << "  ";
      log << it->fiber_pos << "  "
          << "  ";
      log << it->fiber_socket << "  " << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadChamberIndexValuesAnalyzer);
}  // namespace edmtest
