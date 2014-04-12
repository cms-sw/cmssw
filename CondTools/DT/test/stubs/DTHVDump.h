
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edmtest {
  class DTHVDump : public edm::EDAnalyzer
  {
  public:
    explicit  DTHVDump(edm::ParameterSet const& p);
    explicit  DTHVDump(int i) ;
    virtual ~ DTHVDump();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
