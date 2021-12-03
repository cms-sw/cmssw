/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
#include "CondFormats/DataRecord/interface/CSCDBCrosstalkRcd.h"

namespace edmtest {
  class CSCCrossTalkDBReadAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit CSCCrossTalkDBReadAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCCrossTalkDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<CSCDBCrosstalk, CSCDBCrosstalkRcd> token_;
  };

  void CSCCrossTalkDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    std::ofstream DBXtalkFile("dbxtalk.dat", std::ios::out);
    int counter = 0;
    using namespace edm::eventsetup;

    edm::LogInfo log("CSCDBCrosstalk");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCDBCrosstalk* mycrosstalk = &context.getData(token_);
    std::vector<CSCDBCrosstalk::Item>::const_iterator it;
    for (it = mycrosstalk->crosstalk.begin(); it != mycrosstalk->crosstalk.end(); ++it) {
      counter++;
      DBXtalkFile << counter << "  " << it->xtalk_slope_right << "  " << it->xtalk_intercept_right << "  "
                  << it->xtalk_slope_left << "  " << it->xtalk_intercept_left << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCCrossTalkDBReadAnalyzer);
}  // namespace edmtest
