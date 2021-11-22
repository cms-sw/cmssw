
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

namespace edmtest {
  class DTFullMapPrint : public edm::EDAnalyzer {
  public:
    explicit DTFullMapPrint(edm::ParameterSet const& p);
    explicit DTFullMapPrint(int i);
    virtual ~DTFullMapPrint();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ESGetToken<DTReadOutMapping, DTReadOutMappingRcd> es_token;
  };
}  // namespace edmtest
