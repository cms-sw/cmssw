
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edmtest {
  class DTCompactMapDump : public edm::EDAnalyzer {
  public:
    explicit DTCompactMapDump(edm::ParameterSet const& p);
    explicit DTCompactMapDump(int i);
    ~DTCompactMapDump() override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endJob() override;

  private:
    std::string fileName;
  };
}  // namespace edmtest
