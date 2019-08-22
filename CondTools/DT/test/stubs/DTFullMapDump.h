
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
    virtual ~DTFullMapDump();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();

  private:
    std::string fileName;
  };
}  // namespace edmtest
