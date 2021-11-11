
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"

namespace edmtest {
  class DTDeadPrint : public edm::EDAnalyzer {
  public:
    explicit DTDeadPrint(edm::ParameterSet const& p);
    explicit DTDeadPrint(int i);
    virtual ~DTDeadPrint();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ESGetToken<DTDeadFlag, DTDeadFlagRcd> es_token;
  };
}  // namespace edmtest
