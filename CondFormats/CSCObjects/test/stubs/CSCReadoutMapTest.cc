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

#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
#include "CondFormats/DataRecord/interface/CSCCrateMapRcd.h"

#include "CondFormats/CSCObjects/interface/CSCMapItem.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

using namespace std;
namespace edmtest {
  class CSCReadoutMapTest : public edm::EDAnalyzer {
  public:
    explicit CSCReadoutMapTest(edm::ParameterSet const& p) {}
    explicit CSCReadoutMapTest(int i) {}
    ~CSCReadoutMapTest() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCReadoutMapTest::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;

    std::cout << "+===================+" << std::endl;
    std::cout << "| CSCReadoutMapTest |" << std::endl;
    std::cout << "+===================+" << std::endl;

    std::cout << "run " << e.id().run() << std::endl;
    std::cout << "event " << e.id().event() << std::endl;

    edm::ESHandle<CSCCrateMap> hcrate;
    context.get<CSCCrateMapRcd>().get(hcrate);
    const CSCCrateMap* pcrate = hcrate.product();

    const int ncrates = 60;
    const int ndmb = 10;
    const int missing = 6;
    int count = 0;
    int countall = 0;

    for (int jc = 0; jc != ncrates; ++jc) {
      for (int jd = 0; jd != ndmb; ++jd) {
        if (jd + 1 == missing)
          continue;     // there's no dmb=6
        int jcfeb = 0;  // just set something
        ++countall;
        // The iterations here cover more than available chambers so need
        // to skip non-existing ones
        int jdmb = jd + 1;
        int jcrate = jc + 1;

        int cscid = jdmb;
        if (jdmb >= 6)
          --cscid;
        int key = jcrate * 10 + cscid;
        CSCCrateMap::CSCMap::const_iterator it = (pcrate->crate_map).find(key);
        if (it != (pcrate->crate_map).end()) {
          ++count;
          CSCDetId id = pcrate->detId(jcrate, jdmb, jcfeb);  // *** TEST THE detId BUILDER FOR CHAMBER ***
          std::cout << "Built CSCDetId for chamber # " << count << " id= " << id << " count " << countall << std::endl;

          if (id.station() == 1 && id.ring() == 1) {
            jcfeb = 4;                                         // Split off ME1a
            CSCDetId id = pcrate->detId(jcrate, jdmb, jcfeb);  // *** TEST THE detId BUILDER FOR CHAMBER ME1a ***
            std::cout << "Built CSCDetId for ME1a, id= " << id << " count " << countall << std::endl;
          }

          for (int il = 1; il != 7; ++il) {
            CSCDetId id = pcrate->detId(jcrate, jdmb, jcfeb, il);  // *** TEST THE detId BUILDER FOR LAYER ***
            std::cout << "Built CSCDetId for layer # " << il << " id= " << id << std::endl;
          }

        } else {
          std::cout << "no chamber at count = " << countall << std::endl;
        }
      }
    }
    std::cout << "+==========================+" << std::endl;
    std::cout << "| End of CSCReadoutMapTest |" << std::endl;
    std::cout << "+==========================+" << std::endl;

  }  // namespace
  DEFINE_FWK_MODULE(CSCReadoutMapTest);
}  // namespace edmtest
