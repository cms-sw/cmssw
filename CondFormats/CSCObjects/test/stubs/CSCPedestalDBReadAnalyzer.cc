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

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

namespace edmtest {
  class CSCPedestalDBReadAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit CSCPedestalDBReadAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCPedestalDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<CSCDBPedestals, CSCDBPedestalsRcd> token_;
  };

  void CSCPedestalDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    const float epsilon = 1.E-09;  // some 'small' value to test for non-positive values.

    using namespace edm::eventsetup;
    std::ofstream DBPedestalFile("dbpeds.dat", std::ios::out);
    int counter = 0;

    edm::LogSystem log("CSCDBPedestals");

    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCDBPedestals* myped = &context.getData(token_);
    CSCDBPedestals::PedestalContainer::const_iterator it;

    for (it = myped->pedestals.begin(); it != myped->pedestals.end(); ++it) {
      counter++;
      DBPedestalFile << counter << "  " << it->ped << "  " << it->rms << std::endl;
      if (it->rms <= epsilon)
        DBPedestalFile << " ERROR? pedestal width <= " << epsilon << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCPedestalDBReadAnalyzer);
}  // namespace edmtest
