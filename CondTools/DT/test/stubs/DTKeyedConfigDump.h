
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondTools/DT/interface/DTKeyedConfigCache.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"
#include "CondFormats/DataRecord/interface/DTKeyedConfigListRcd.h"

#include <string>

namespace edmtest {
  class DTKeyedConfigDump : public edm::EDAnalyzer {
  public:
    explicit DTKeyedConfigDump(edm::ParameterSet const& p);

    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    const bool dumpCCBKeys;
    const bool dumpAllData;
    DTKeyedConfigCache cfgCache;
    const edm::ESGetToken<DTCCBConfig, DTCCBConfigRcd> configToken_;
    edm::ESGetToken<cond::persistency::KeyList, DTKeyedConfigListRcd> keyListToken_;
  };
}  // namespace edmtest
