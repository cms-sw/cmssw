#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"

//
// class declaration
//

class CSCMap1Read : public edm::one::EDAnalyzer<> {
public:
  explicit CSCMap1Read(const edm::ParameterSet &);
  ~CSCMap1Read() override = default;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCMap1Read);

CSCMap1Read::CSCMap1Read(const edm::ParameterSet &) {}

void CSCMap1Read::analyze(const edm::Event &, const edm::EventSetup &) {
  CSCMapItem::MapItem item;
  cscmap1 *map = new cscmap1();
  std::cout << " Connected cscmap ... " << std::endl;

  // Get information by chamber ID.
  int chamberid = 122090;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Method chamberid, input: chamberID  " << chamberid << std::endl;
  std::cout << std::endl;
  map->chamber(chamberid, &item);

  std::cout << "cscLabel  "
            << "  " << item.chamberLabel << std::endl;
  std::cout << "cscId  "
            << "  " << item.chamberId << std::endl;
  std::cout << "endcap  "
            << "  " << item.endcap << std::endl;
  std::cout << "station  "
            << "  " << item.station << std::endl;
  std::cout << "ring  "
            << "  " << item.ring << std::endl;
  std::cout << "chamber  "
            << "  " << item.chamber << std::endl;
  std::cout << "cscIndex  "
            << "  " << item.cscIndex << std::endl;
  std::cout << "layerIndex  "
            << "  " << item.layerIndex << std::endl;
  std::cout << "stripIndex  "
            << "  " << item.stripIndex << std::endl;
  std::cout << "anodeIndex  "
            << "  " << item.anodeIndex << std::endl;
  std::cout << "strips  "
            << "  " << item.strips << std::endl;
  std::cout << "anodes  "
            << "  " << item.anodes << std::endl;
  std::cout << "crateLabel  "
            << "  " << item.crateLabel << std::endl;
  std::cout << "crateid  "
            << "  " << item.crateid << std::endl;
  std::cout << "sector  "
            << "  " << item.sector << std::endl;
  std::cout << "trig_sector  "
            << "  " << item.trig_sector << std::endl;
  std::cout << "dmb  "
            << "  " << item.dmb << std::endl;
  std::cout << "cscid  "
            << "  " << item.cscid << std::endl;
  std::cout << "ddu  "
            << "  " << item.ddu << std::endl;
  std::cout << "ddu_input  "
            << "  " << item.ddu_input << std::endl;
  std::cout << "slink  "
            << "  " << item.slink << std::endl;
  std::cout << "fed_crate  "
            << "  " << item.fed_crate << std::endl;
  std::cout << "ddu_slot  "
            << "  " << item.ddu_slot << std::endl;
  std::cout << "dcc_fifo  "
            << "  " << item.dcc_fifo << std::endl;
  std::cout << "fiber_crate  "
            << "  " << item.fiber_crate << std::endl;
  std::cout << "fiber_pos  "
            << "  " << item.fiber_pos << std::endl;
  std::cout << "fiber_socket  "
            << "  " << item.fiber_socket << std::endl;

  // Get information by crateid and dmb.
  int crateid = 33;
  int dmb = 7;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Method cratedmb, input: crateid " << crateid << ", dmb " << dmb << std::endl;
  std::cout << std::endl;
  map->cratedmb(crateid, dmb, &item);

  std::cout << "cscLabel  "
            << "  " << item.chamberLabel << std::endl;
  std::cout << "cscId  "
            << "  " << item.chamberId << std::endl;
  std::cout << "endcap  "
            << "  " << item.endcap << std::endl;
  std::cout << "station  "
            << "  " << item.station << std::endl;
  std::cout << "ring  "
            << "  " << item.ring << std::endl;
  std::cout << "chamber  "
            << "  " << item.chamber << std::endl;
  std::cout << "cscIndex  "
            << "  " << item.cscIndex << std::endl;
  std::cout << "layerIndex  "
            << "  " << item.layerIndex << std::endl;
  std::cout << "stripIndex  "
            << "  " << item.stripIndex << std::endl;
  std::cout << "anodeIndex  "
            << "  " << item.anodeIndex << std::endl;
  std::cout << "strips  "
            << "  " << item.strips << std::endl;
  std::cout << "anodes  "
            << "  " << item.anodes << std::endl;
  std::cout << "crateLabel  "
            << "  " << item.crateLabel << std::endl;
  std::cout << "crateid  "
            << "  " << item.crateid << std::endl;
  std::cout << "sector  "
            << "  " << item.sector << std::endl;
  std::cout << "trig_sector  "
            << "  " << item.trig_sector << std::endl;
  std::cout << "dmb  "
            << "  " << item.dmb << std::endl;
  std::cout << "cscid  "
            << "  " << item.cscid << std::endl;
  std::cout << "ddu  "
            << "  " << item.ddu << std::endl;
  std::cout << "ddu_input  "
            << "  " << item.ddu_input << std::endl;
  std::cout << "slink  "
            << "  " << item.slink << std::endl;
  std::cout << "fed_crate  "
            << "  " << item.fed_crate << std::endl;
  std::cout << "ddu_slot  "
            << "  " << item.ddu_slot << std::endl;
  std::cout << "dcc_fifo  "
            << "  " << item.dcc_fifo << std::endl;
  std::cout << "fiber_crate  "
            << "  " << item.fiber_crate << std::endl;
  std::cout << "fiber_pos  "
            << "  " << item.fiber_pos << std::endl;
  std::cout << "fiber_socket  "
            << "  " << item.fiber_socket << std::endl;

  // Get information by rui and ddu_input.
  int rui = 2;
  int ddu_input = 2;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Method ruiddu, input: rui " << rui << ", ddu_input " << ddu_input << std::endl;
  std::cout << std::endl;
  map->ruiddu(rui, ddu_input, &item);

  std::cout << "cscLabel  "
            << "  " << item.chamberLabel << std::endl;
  std::cout << "cscId  "
            << "  " << item.chamberId << std::endl;
  std::cout << "endcap  "
            << "  " << item.endcap << std::endl;
  std::cout << "station  "
            << "  " << item.station << std::endl;
  std::cout << "ring  "
            << "  " << item.ring << std::endl;
  std::cout << "chamber  "
            << "  " << item.chamber << std::endl;
  std::cout << "cscIndex  "
            << "  " << item.cscIndex << std::endl;
  std::cout << "layerIndex  "
            << "  " << item.layerIndex << std::endl;
  std::cout << "stripIndex  "
            << "  " << item.stripIndex << std::endl;
  std::cout << "anodeIndex  "
            << "  " << item.anodeIndex << std::endl;
  std::cout << "strips  "
            << "  " << item.strips << std::endl;
  std::cout << "anodes  "
            << "  " << item.anodes << std::endl;
  std::cout << "crateLabel  "
            << "  " << item.crateLabel << std::endl;
  std::cout << "crateid  "
            << "  " << item.crateid << std::endl;
  std::cout << "sector  "
            << "  " << item.sector << std::endl;
  std::cout << "trig_sector  "
            << "  " << item.trig_sector << std::endl;
  std::cout << "dmb  "
            << "  " << item.dmb << std::endl;
  std::cout << "cscid  "
            << "  " << item.cscid << std::endl;
  std::cout << "ddu  "
            << "  " << item.ddu << std::endl;
  std::cout << "ddu_input  "
            << "  " << item.ddu_input << std::endl;
  std::cout << "slink  "
            << "  " << item.slink << std::endl;
  std::cout << "fed_crate  "
            << "  " << item.fed_crate << std::endl;
  std::cout << "ddu_slot  "
            << "  " << item.ddu_slot << std::endl;
  std::cout << "dcc_fifo  "
            << "  " << item.dcc_fifo << std::endl;
  std::cout << "fiber_crate  "
            << "  " << item.fiber_crate << std::endl;
  std::cout << "fiber_pos  "
            << "  " << item.fiber_pos << std::endl;
  std::cout << "fiber_socket  "
            << "  " << item.fiber_socket << std::endl;
}
void CSCMap1Read::beginJob() {
  std::cout << "Here is the start" << std::endl;
  std::cout << "-----------------" << std::endl;
}
void CSCMap1Read::endJob() {
  std::cout << "---------------" << std::endl;
  std::cout << "Here is the end" << std::endl;
}
