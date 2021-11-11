
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DataRecord/interface/DTRangeT0Rcd.h"

namespace edmtest {
  class DTRangeT0Print : public edm::EDAnalyzer {
  public:
    explicit DTRangeT0Print(edm::ParameterSet const& p);
    explicit DTRangeT0Print(int i);
    virtual ~DTRangeT0Print();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ESGetToken<DTRangeT0, DTRangeT0Rcd> es_token;
  };
}  // namespace edmtest
