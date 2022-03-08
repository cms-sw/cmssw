
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

namespace edmtest {
  class DTT0Print : public edm::one::EDAnalyzer<> {
  public:
    explicit DTT0Print(edm::ParameterSet const& p);
    explicit DTT0Print(int i);
    ~DTT0Print() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<DTT0, DTT0Rcd> es_token;
  };
}  // namespace edmtest
