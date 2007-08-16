
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTDeadWrite : public edm::EDAnalyzer
  {
  public:
    explicit  DTDeadWrite(edm::ParameterSet const& p);
    explicit  DTDeadWrite(int i) ;
    virtual ~ DTDeadWrite();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
  private:
  };
}
