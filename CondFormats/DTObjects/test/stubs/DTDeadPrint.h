
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

namespace edmtest {
  class DTDeadPrint : public edm::one::EDAnalyzer<> {
  public:
    explicit DTDeadPrint(edm::ParameterSet const& p);
    explicit DTDeadPrint(int i);
    ~DTDeadPrint() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<DTDeadFlag, DTDeadFlagRcd> es_token;
  };
}  // namespace edmtest
