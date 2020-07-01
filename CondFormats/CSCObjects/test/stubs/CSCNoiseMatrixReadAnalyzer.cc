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

#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"

using namespace std;

namespace edmtest {
  class CSCNoiseMatrixReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCNoiseMatrixReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCNoiseMatrixReadAnalyzer(int i) {}
    ~CSCNoiseMatrixReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCNoiseMatrixReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCNoiseMatrix> pNoiseMatrix;
    context.get<CSCNoiseMatrixRcd>().get(pNoiseMatrix);

    const CSCNoiseMatrix* myNoiseMatrix = pNoiseMatrix.product();
    std::map<int, std::vector<CSCNoiseMatrix::Item> >::const_iterator it;
    for (it = myNoiseMatrix->matrix.begin(); it != myNoiseMatrix->matrix.end(); ++it) {
      std::cout << "layer id found " << it->first << std::endl;
      std::vector<CSCNoiseMatrix::Item>::const_iterator matrixit;
      for (matrixit = it->second.begin(); matrixit != it->second.end(); ++matrixit) {
        std::cout << "  matrix elem33:  " << matrixit->elem33 << " matrix elem34: " << matrixit->elem34 << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCNoiseMatrixReadAnalyzer);
}  // namespace edmtest
