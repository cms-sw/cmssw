
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace edmtest {
  class DTMtimePrint : public edm::EDAnalyzer
  {
  public:
    explicit  DTMtimePrint(edm::ParameterSet const& p);
    explicit  DTMtimePrint(int i) ;
    virtual ~ DTMtimePrint();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
