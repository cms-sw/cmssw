
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTTtrigPrint : public edm::EDAnalyzer
  {
  public:
    explicit  DTTtrigPrint(edm::ParameterSet const& p);
    explicit  DTTtrigPrint(int i) ;
    virtual ~ DTTtrigPrint();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
