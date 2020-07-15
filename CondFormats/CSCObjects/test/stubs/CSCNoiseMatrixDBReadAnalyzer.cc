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

#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCDBNoiseMatrixRcd.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"

namespace edmtest {
  class CSCNoiseMatrixDBReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCNoiseMatrixDBReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCNoiseMatrixDBReadAnalyzer(int i) {}
    ~CSCNoiseMatrixDBReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCNoiseMatrixDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    std::ofstream DBNoiseMatrixFile("dbmatrix.dat", std::ios::out);
    int counter = 0;
    using namespace edm::eventsetup;
    std::cout << " Run# " << e.id().run() << std::endl;
    std::cout << " Event# " << e.id().event() << std::endl;
    std::cout << " Matrix values are written to file dbmatrix.dat & errors are written to cerr." << std::endl;
    edm::ESHandle<CSCDBNoiseMatrix> pNoiseMatrix;
    context.get<CSCDBNoiseMatrixRcd>().get(pNoiseMatrix);

    const CSCDBNoiseMatrix* myNoiseMatrix = pNoiseMatrix.product();
    std::cout << " Scale factor for conversion to int was " << myNoiseMatrix->factor_noise << std::endl;

    CSCIndexer indexer;

    std::vector<CSCDBNoiseMatrix::Item>::const_iterator it;

    for (it = myNoiseMatrix->matrix.begin(); it != myNoiseMatrix->matrix.end(); ++it) {
      ++counter;
      std::pair<CSCDetId, CSCIndexer::IndexType> thePair = indexer.detIdFromStripChannelIndex(counter);
      DBNoiseMatrixFile << counter << "  " << thePair.first << " chan " << thePair.second << "  " << it->elem33 << "  "
                        << it->elem34 << "  " << it->elem44 << "  " << it->elem35 << "  " << it->elem45 << "  "
                        << it->elem55 << "  " << it->elem46 << "  " << it->elem56 << "  " << it->elem66 << "  "
                        << it->elem57 << "  " << it->elem67 << "  " << it->elem77 << std::endl;
      if (it->elem33 < 0) {
        std::cerr << " 33(1) negative: " << it->elem33 << std::endl;
      }
      if (it->elem44 < 0) {
        std::cerr << " 44(3) negative: " << it->elem44 << std::endl;
      }
      if (it->elem55 < 0) {
        std::cerr << " 55(6) negative: " << it->elem55 << std::endl;
      }
      if (it->elem66 < 0) {
        std::cerr << " 66(9) negative: " << it->elem66 << std::endl;
      }
      if (it->elem77 < 0) {
        std::cerr << " 77(12) negative: " << it->elem77 << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCNoiseMatrixDBReadAnalyzer);
}  // namespace edmtest
