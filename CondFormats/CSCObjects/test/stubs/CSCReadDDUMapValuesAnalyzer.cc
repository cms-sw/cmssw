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

#include "CondFormats/CSCObjects/interface/CSCDDUMap.h"
#include "CondFormats/DataRecord/interface/CSCDDUMapRcd.h"
#include "CondFormats/CSCObjects/interface/CSCMapItem.h"

using namespace std;

namespace edmtest {
  class CSCReadDDUMapValuesAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCReadDDUMapValuesAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCReadDDUMapValuesAnalyzer(int i) {}
    ~CSCReadDDUMapValuesAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCReadDDUMapValuesAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCDDUMap> pDDUMap;
    context.get<CSCDDUMapRcd>().get(pDDUMap);

    const CSCDDUMap* myDDUMap = pDDUMap.product();

    std::map<int, CSCMapItem::MapItem>::const_iterator it;

    int count = 0;
    for (it = myDDUMap->ddu_map.begin(); it != myDDUMap->ddu_map.end(); ++it) {
      count = count + 1;
      std::cout << "Key: ddu_crate*10+ddu_input " << it->first << std::endl;

      std::cout << count << ") ";
      std::cout << it->second.chamberLabel << "  ";
      std::cout << it->second.chamberId << "  ";
      std::cout << it->second.endcap << "  ";
      std::cout << it->second.station << "  ";
      std::cout << it->second.ring << "  ";
      std::cout << it->second.chamber << "  ";
      std::cout << it->second.cscIndex << "  ";
      std::cout << it->second.layerIndex << "  ";
      std::cout << it->second.stripIndex << "  ";
      std::cout << it->second.anodeIndex << "  ";
      std::cout << it->second.strips << "  ";
      std::cout << it->second.anodes << "  ";
      std::cout << it->second.crateLabel << "  ";
      std::cout << it->second.crateid << "  ";
      std::cout << it->second.sector << "  ";
      std::cout << it->second.trig_sector << "  ";
      std::cout << it->second.dmb << "  ";
      std::cout << it->second.cscid << "  ";
      std::cout << it->second.ddu << "  ";
      std::cout << it->second.ddu_input << "  ";
      std::cout << it->second.slink << "  ";
      std::cout << it->second.fed_crate << "  "
                << "  ";
      std::cout << it->second.ddu_slot << "  "
                << "  ";
      std::cout << it->second.dcc_fifo << "  "
                << "  ";
      std::cout << it->second.fiber_crate << "  "
                << "  ";
      std::cout << it->second.fiber_pos << "  "
                << "  ";
      std::cout << it->second.fiber_socket << "  " << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCReadDDUMapValuesAnalyzer);
}  // namespace edmtest
