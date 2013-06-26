
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTMtime;

namespace edmtest {
  class DTMtimeWrite : public edm::EDAnalyzer
  {
  public:
    explicit  DTMtimeWrite(edm::ParameterSet const& p);
    explicit  DTMtimeWrite(int i) ;
    virtual ~ DTMtimeWrite();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
    virtual void chkData( DTMtime* mTime );
  private:
  };
}
