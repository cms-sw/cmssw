
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
class DTGeometry;
class MuonGeometryRecord;

namespace edmtest {
  class DTGeometryDump : public edm::EDAnalyzer {
  public:
    explicit DTGeometryDump(edm::ParameterSet const& p);
    explicit DTGeometryDump(int i);
    virtual ~DTGeometryDump();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
    edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtgeomToken_;
  };
}  // namespace edmtest
