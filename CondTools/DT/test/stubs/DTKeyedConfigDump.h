
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondTools/DT/interface/DTKeyedConfigCache.h"

#include <string>

namespace edmtest {
  class DTKeyedConfigDump : public edm::EDAnalyzer
  {
  public:
    explicit  DTKeyedConfigDump(edm::ParameterSet const& p);
    explicit  DTKeyedConfigDump(int i) ;
    virtual ~ DTKeyedConfigDump();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
    bool dumpCCBKeys;
    bool dumpAllData;
    DTKeyedConfigCache cfgCache;
  };
}
