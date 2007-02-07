
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTRangeT0Write : public edm::EDAnalyzer
  {
  public:
    explicit  DTRangeT0Write(edm::ParameterSet const& p);
    explicit  DTRangeT0Write(int i) ;
    virtual ~ DTRangeT0Write();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
  private:
  };
}
