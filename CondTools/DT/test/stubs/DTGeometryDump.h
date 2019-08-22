
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edmtest {
  class DTGeometryDump : public edm::EDAnalyzer {
  public:
    explicit DTGeometryDump(edm::ParameterSet const& p);
    explicit DTGeometryDump(int i);
    virtual ~DTGeometryDump();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

  private:
  };
}  // namespace edmtest
