
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTDeadFlag;

namespace edmtest {
  class DTDeadUpdate : public edm::EDAnalyzer
  {
  public:
    explicit  DTDeadUpdate(edm::ParameterSet const& p);
    explicit  DTDeadUpdate(int i) ;
    virtual ~ DTDeadUpdate();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
  private:
    DTDeadFlag* dSum;
  };
}
