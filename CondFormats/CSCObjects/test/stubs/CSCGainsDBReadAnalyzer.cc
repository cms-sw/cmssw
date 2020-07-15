/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
#include "CondFormats/DataRecord/interface/CSCDBGainsRcd.h"

namespace edmtest {
  class CSCGainsDBReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCGainsDBReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCGainsDBReadAnalyzer(int i) {}
    ~CSCGainsDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCGainsDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    std::ofstream DBGainsFile("dbgains.dat", std::ios::out);
    int counter = 0;
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCDBGains> pGains;
    context.get<CSCDBGainsRcd>().get(pGains);

    const CSCDBGains* mygains = pGains.product();
    std::vector<CSCDBGains::Item>::const_iterator it;

    for (it = mygains->gains.begin(); it != mygains->gains.end(); ++it) {
      counter++;
      DBGainsFile << counter << "  " << it->gain_slope << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCGainsDBReadAnalyzer);
}  // namespace edmtest
