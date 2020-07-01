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

#include "CondFormats/CSCObjects/interface/CSCDBChipSpeedCorrection.h"
#include "CondFormats/DataRecord/interface/CSCDBChipSpeedCorrectionRcd.h"

namespace edmtest {
  class CSCChipSpeedCorrectionDBReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCChipSpeedCorrectionDBReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCChipSpeedCorrectionDBReadAnalyzer(int i) {}
    ~CSCChipSpeedCorrectionDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCChipSpeedCorrectionDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    //const float epsilon = 1.E-09; // some 'small' value to test for non-positive values.
    //*    const float epsilon = 20; // some 'small' value to test

    using namespace edm::eventsetup;
    std::ofstream DBChipSpeedCorrectionFile("dbChipSpeedCorrection.dat", std::ios::out);
    int counter = 0;

    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCDBChipSpeedCorrection> pChipCorr;
    context.get<CSCDBChipSpeedCorrectionRcd>().get(pChipCorr);

    const CSCDBChipSpeedCorrection* myChipCorr = pChipCorr.product();
    CSCDBChipSpeedCorrection::ChipSpeedContainer::const_iterator it;

    for (it = myChipCorr->chipSpeedCorr.begin(); it != myChipCorr->chipSpeedCorr.end(); ++it) {
      counter++;
      DBChipSpeedCorrectionFile << counter << "  " << it->speedCorr / 100. << std::endl;
      //* if ( it->speedCorr <= epsilon ) DBChipSpeedCorrectionFile << " ERROR? Chip Correction <= " << epsilon << std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCChipSpeedCorrectionDBReadAnalyzer);
}  // namespace edmtest
