
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
class DTGeometry;
class MuonGeometryRecord;

namespace edmtest {
  class DTGeometryDump : public edm::one::EDAnalyzer<> {
  public:
    explicit DTGeometryDump(edm::ParameterSet const& p);
    explicit DTGeometryDump(int i);
    ~DTGeometryDump() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtgeomToken_;
  };
}  // namespace edmtest
