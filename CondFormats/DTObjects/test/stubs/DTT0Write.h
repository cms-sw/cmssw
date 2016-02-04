
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTT0Write : public edm::EDAnalyzer
  {
  public:
    explicit  DTT0Write(edm::ParameterSet const& p);
    explicit  DTT0Write(int i) ;
    virtual ~ DTT0Write();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
  private:
  };
}
