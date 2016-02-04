
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTTtrig;

namespace edmtest {
  class DTTtrigWrite : public edm::EDAnalyzer
  {
  public:
    explicit  DTTtrigWrite(edm::ParameterSet const& p);
    explicit  DTTtrigWrite(int i) ;
    virtual ~ DTTtrigWrite();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
    virtual void chkData( DTTtrig* tTrig );
  private:
  };
}
