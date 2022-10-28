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

#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/DataRecord/interface/CSCNoiseMatrixRcd.h"

using namespace std;

namespace edmtest {
  class CSCNoiseMatrixReadAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit CSCNoiseMatrixReadAnalyzer(edm::ParameterSet const& p) : noiseToken_{esConsumes()} {}
    ~CSCNoiseMatrixReadAnalyzer() override {}
    void analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& c) const override;

  private:
    edm::ESGetToken<CSCNoiseMatrix, CSCNoiseMatrixRcd> noiseToken_;
  };

  void CSCNoiseMatrixReadAnalyzer::analyze(edm::StreamID, const edm::Event& e, const edm::EventSetup& context) const {
    using namespace edm::eventsetup;
    edm::LogSystem log("CSCNoiseMatrix");
    log << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    log << " ---EVENT NUMBER " << e.id().event() << std::endl;

    const CSCNoiseMatrix* myNoiseMatrix = &context.getData(noiseToken_);
    std::map<int, std::vector<CSCNoiseMatrix::Item> >::const_iterator it;
    for (it = myNoiseMatrix->matrix.begin(); it != myNoiseMatrix->matrix.end(); ++it) {
      log << "layer id found " << it->first << std::endl;
      std::vector<CSCNoiseMatrix::Item>::const_iterator matrixit;
      for (matrixit = it->second.begin(); matrixit != it->second.end(); ++matrixit) {
        log << "  matrix elem33:  " << matrixit->elem33 << " matrix elem34: " << matrixit->elem34 << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCNoiseMatrixReadAnalyzer);
}  // namespace edmtest
