/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <fstream>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"

namespace edmtest {
  class CSCChipSpeedCorrectionDBReadAnalyzer : public edm::one::EDAnalyzer<> {
  public:
    explicit CSCChipSpeedCorrectionDBReadAnalyzer(edm::ParameterSet const& p) : token_{esConsumes()} {}
    ~CSCChipSpeedCorrectionDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<CSCDBChipSpeedCorrection, CSCDBChipSpeedCorrectionRcd> token_;
  };

  void CSCChipSpeedCorrectionDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    //const float epsilon = 1.E-09; // some 'small' value to test for non-positive values.
    //*    const float epsilon = 20; // some 'small' value to test

    using namespace edm::eventsetup;
    std::ofstream DBChipSpeedCorrectionFile("dbChipSpeedCorrection.dat", std::ios::out);
    int counter = 0;

    edm::LogSystem log("CSCDBChipSpeedCorrection");

    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCDBChipSpeedCorrection* myChipCorr = &context.getData(token_);
    CSCDBChipSpeedCorrection::ChipSpeedContainer::const_iterator it;

    for (it = myChipCorr->chipSpeedCorr.begin(); it != myChipCorr->chipSpeedCorr.end(); ++it) {
      counter++;
      DBChipSpeedCorrectionFile << counter << "  " << it->speedCorr / 100. << std::endl;
      //* if ( it->speedCorr <= epsilon ) DBChipSpeedCorrectionFile << " ERROR? Chip Correction <= " << epsilon << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCChipSpeedCorrectionDBReadAnalyzer);
}  // namespace edmtest
