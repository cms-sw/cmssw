
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

namespace edmtest {
  class DTTtrigPrint : public edm::one::EDAnalyzer<> {
  public:
    explicit DTTtrigPrint(edm::ParameterSet const& p);
    explicit DTTtrigPrint(int i);
    ~DTTtrigPrint() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<DTTtrig, DTTtrigRcd> es_token;
  };
}  // namespace edmtest
