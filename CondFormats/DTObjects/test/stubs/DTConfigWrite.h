
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest {
  class DTConfigWrite : public edm::EDAnalyzer {
  public:
    explicit DTConfigWrite(edm::ParameterSet const& p);
    explicit DTConfigWrite(int i);
    ~DTConfigWrite() override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endJob() override;

  private:
  };
}  // namespace edmtest
