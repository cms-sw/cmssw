
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
class DTHVStatus;
class DTHVStatusRcd;
namespace edmtest {
  class DTHVDump : public edm::one::EDAnalyzer<> {
  public:
    explicit DTHVDump(edm::ParameterSet const& p);
    explicit DTHVDump(int i);
    ~DTHVDump() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
    edm::ESGetToken<DTHVStatus, DTHVStatusRcd> dthvstatusToken_;
  };
}  // namespace edmtest
