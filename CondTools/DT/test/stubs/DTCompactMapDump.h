
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edmtest {
  class DTCompactMapDump : public edm::EDAnalyzer
  {
  public:
    explicit  DTCompactMapDump(edm::ParameterSet const& p);
    explicit  DTCompactMapDump(int i) ;
    virtual ~ DTCompactMapDump();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
  private:
    std::string fileName;
  };
}
