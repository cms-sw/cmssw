
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTMapWrite : public edm::one::EDAnalyzer<> {
  public:
    explicit DTMapWrite(edm::ParameterSet const& p);
    explicit DTMapWrite(int i);
    ~DTMapWrite() override = default;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endJob() override;

  private:
  };
}  // namespace edmtest
