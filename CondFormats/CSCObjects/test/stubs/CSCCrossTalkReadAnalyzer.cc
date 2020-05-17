/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"

using namespace std;

namespace edmtest {
  class CSCCrossTalkReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCCrossTalkReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCCrossTalkReadAnalyzer(int i) {}
    ~CSCCrossTalkReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCCrossTalkReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCcrosstalk> pcrosstalk;
    context.get<CSCcrosstalkRcd>().get(pcrosstalk);

    const CSCcrosstalk* mycrosstalk = pcrosstalk.product();
    std::map<int, std::vector<CSCcrosstalk::Item> >::const_iterator it;
    for (it = mycrosstalk->crosstalk.begin(); it != mycrosstalk->crosstalk.end(); ++it) {
      std::cout << "layer id found " << it->first << std::endl;
      std::vector<CSCcrosstalk::Item>::const_iterator crosstalkit;
      for (crosstalkit = it->second.begin(); crosstalkit != it->second.end(); ++crosstalkit) {
        std::cout << "  crosstalk_slope_right:  " << crosstalkit->xtalk_slope_right
                  << " crosstalk_intercept_right: " << crosstalkit->xtalk_intercept_right << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCCrossTalkReadAnalyzer);
}  // namespace edmtest
