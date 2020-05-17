
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTMapWrite : public edm::EDAnalyzer {
  public:
    explicit DTMapWrite(edm::ParameterSet const& p);
    explicit DTMapWrite(int i);
    ~DTMapWrite() override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endJob() override;

  private:
  };
}  // namespace edmtest
