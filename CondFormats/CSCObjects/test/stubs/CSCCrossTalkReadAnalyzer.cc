/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCcrosstalkRcd.h"

using namespace std;

namespace edmtest {
  class CSCCrossTalkReadAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCCrossTalkReadAnalyzer(edm::ParameterSet const& p) : crosstalkToken_{esConsumes()} {}
    ~CSCCrossTalkReadAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCcrosstalk, CSCcrosstalkRcd> crosstalkToken_;
  };

  void CSCCrossTalkReadAnalyzer::analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& context) const {
    using namespace edm::eventsetup;
    // Context is not used.
    edm::LogSystem log("CSCCrossTalk");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCcrosstalk* mycrosstalk = &context.getData(crosstalkToken_);
    std::map<int, std::vector<CSCcrosstalk::Item> >::const_iterator it;
    for (it = mycrosstalk->crosstalk.begin(); it != mycrosstalk->crosstalk.end(); ++it) {
      log << "layer id found " << it->first << std::endl;
      std::vector<CSCcrosstalk::Item>::const_iterator crosstalkit;
      for (crosstalkit = it->second.begin(); crosstalkit != it->second.end(); ++crosstalkit) {
        log << "  crosstalk_slope_right:  " << crosstalkit->xtalk_slope_right
            << " crosstalk_intercept_right: " << crosstalkit->xtalk_intercept_right << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCCrossTalkReadAnalyzer);
}  // namespace edmtest
