
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "CondFormats/DTObjects/interface/DTConfigList.h"
//#include "CondTools/DT/interface/DTConfigHandler.h"
//#include "CondTools/DT/interface/DTDBSession.h"

#include <string>

namespace edmtest {
  class DTConfigPrint : public edm::EDAnalyzer
  {
  public:
    explicit  DTConfigPrint(edm::ParameterSet const& p);
    explicit  DTConfigPrint(int i) ;
    virtual ~ DTConfigPrint();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
    std::string connect;
    std::string auth_path;
    std::string catalog;
    std::string token;
    bool local;
//    DTDBSession* session;
//    const DTConfigList* rs;
//    DTConfigHandler* ri;
  };
}
