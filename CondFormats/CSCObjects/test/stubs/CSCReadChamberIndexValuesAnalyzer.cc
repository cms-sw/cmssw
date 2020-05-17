/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
#include "CondFormats/DataRecord/interface/CSCChamberIndexRcd.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"

using namespace std;

namespace edmtest {
  class CSCReadChamberIndexValuesAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadChamberIndexValuesAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCReadChamberIndexValuesAnalyzer(int i) {}
    ~CSCReadChamberIndexValuesAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCReadChamberIndexValuesAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCChamberIndex> pChIndex;
    context.get<CSCChamberIndexRcd>().get(pChIndex);

    const CSCChamberIndex* myChIndex = pChIndex.product();

    std::vector<CSCMapItem::MapItem>::const_iterator it;

    int count = 0;
    for (it = myChIndex->ch_index.begin(); it != myChIndex->ch_index.end(); ++it) {
      count++;

      std::cout << count << ") ";
      std::cout << it->chamberLabel << "  ";
      std::cout << it->chamberId << "  ";
      std::cout << it->endcap << "  ";
      std::cout << it->station << "  ";
      std::cout << it->ring << "  ";
      std::cout << it->chamber << "  ";
      std::cout << it->cscIndex << "  ";
      std::cout << it->layerIndex << "  ";
      std::cout << it->stripIndex << "  ";
      std::cout << it->anodeIndex << "  ";
      std::cout << it->strips << "  ";
      std::cout << it->anodes << "  ";
      std::cout << it->crateLabel << "  ";
      std::cout << it->crateid << "  ";
      std::cout << it->sector << "  ";
      std::cout << it->trig_sector << "  ";
      std::cout << it->dmb << "  ";
      std::cout << it->cscid << "  ";
      std::cout << it->ddu << "  ";
      std::cout << it->ddu_input << "  ";
      std::cout << it->slink << "  ";
      std::cout << it->fed_crate << "  "
                << "  ";
      std::cout << it->ddu_slot << "  "
                << "  ";
      std::cout << it->dcc_fifo << "  "
                << "  ";
      std::cout << it->fiber_crate << "  "
                << "  ";
      std::cout << it->fiber_pos << "  "
                << "  ";
      std::cout << it->fiber_socket << "  " << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadChamberIndexValuesAnalyzer);
}  // namespace edmtest
