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

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"

using namespace std;

namespace edmtest {
  class CSCGainsReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCGainsReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCGainsReadAnalyzer(int i) {}
    ~CSCGainsReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCGainsReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCGains> pGains;
    context.get<CSCGainsRcd>().get(pGains);

    const CSCGains* mygains = pGains.product();
    std::map<int, std::vector<CSCGains::Item> >::const_iterator it;
    for (it = mygains->gains.begin(); it != mygains->gains.end(); ++it) {
      std::cout << "layer id found " << it->first << std::endl;
      std::vector<CSCGains::Item>::const_iterator gainsit;
      for (gainsit = it->second.begin(); gainsit != it->second.end(); ++gainsit) {
        std::cout << "  gains:  " << gainsit->gain_slope << " intercept: " << gainsit->gain_intercept << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCGainsReadAnalyzer);
}  // namespace edmtest
