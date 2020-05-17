
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edmtest {
  class DTFullMapDump : public edm::EDAnalyzer {
  public:
    explicit DTFullMapDump(edm::ParameterSet const& p);
    explicit DTFullMapDump(int i);
    ~DTFullMapDump() override;
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;
    void endJob() override;

  private:
    std::string fileName;
  };
}  // namespace edmtest
