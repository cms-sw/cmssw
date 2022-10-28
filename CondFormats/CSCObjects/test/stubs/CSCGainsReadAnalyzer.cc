/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/DataRecord/interface/CSCGainsRcd.h"

using namespace std;

namespace edmtest {
  class CSCGainsReadAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCGainsReadAnalyzer(edm::ParameterSet const& p) : gainsToken_{esConsumes()} {}
    ~CSCGainsReadAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCGains, CSCGainsRcd> gainsToken_;
  };

  void CSCGainsReadAnalyzer::analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& context) const {
    using namespace edm::eventsetup;

    edm::LogSystem log("CSCGains");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCGains* mygains = &context.getData(gainsToken_);
    std::map<int, std::vector<CSCGains::Item> >::const_iterator it;
    for (it = mygains->gains.begin(); it != mygains->gains.end(); ++it) {
      log << "layer id found " << it->first << std::endl;
      std::vector<CSCGains::Item>::const_iterator gainsit;
      for (gainsit = it->second.begin(); gainsit != it->second.end(); ++gainsit) {
        log << "  gains:  " << gainsit->gain_slope << " intercept: " << gainsit->gain_intercept << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCGainsReadAnalyzer);
}  // namespace edmtest
