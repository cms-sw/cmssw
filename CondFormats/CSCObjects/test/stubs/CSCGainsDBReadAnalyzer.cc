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

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"

namespace edmtest {
  class CSCGainsDBReadAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit CSCGainsDBReadAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCGainsDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<CSCDBGains, CSCDBGainsRcd> token_;
  };

  void CSCGainsDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    std::ofstream DBGainsFile("dbgains.dat", std::ios::out);
    int counter = 0;

    edm::LogInfo log("CSCDBGains");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCDBGains* mygains = &context.getData(token_);
    std::vector<CSCDBGains::Item>::const_iterator it;

    for (it = mygains->gains.begin(); it != mygains->gains.end(); ++it) {
      counter++;
      DBGainsFile << counter << "  " << it->gain_slope << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCGainsDBReadAnalyzer);
}  // namespace edmtest
