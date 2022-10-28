
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DataRecord/interface/DTCCBConfigRcd.h"

//#include "CondFormats/DTObjects/interface/DTConfigList.h"
//#include "CondTools/DT/interface/DTConfigHandler.h"
//#include "CondTools/DT/interface/DTDBSession.h"

#include <string>

namespace edmtest {
  class DTConfigPrint : public edm::one::EDAnalyzer<> {
  public:
    explicit DTConfigPrint(edm::ParameterSet const& p);
    explicit DTConfigPrint(int i);
    ~DTConfigPrint() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    std::string connect;
    std::string auth_path;
    std::string catalog;
    std::string token;
    bool local;
    edm::ESGetToken<DTCCBConfig, DTCCBConfigRcd> es_token;
    //    DTDBSession* session;
    //    const DTConfigList* rs;
    //    DTConfigHandler* ri;
  };
}  // namespace edmtest
