
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTDeadFlag;

namespace edmtest {
  class DTDeadUpdate : public edm::EDAnalyzer
  {
  public:
    explicit  DTDeadUpdate(edm::ParameterSet const& p);
    explicit  DTDeadUpdate(int i) ;
    virtual ~ DTDeadUpdate();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
    virtual void endJob();
  private:
    void fill_dead_HV( const char* file, DTDeadFlag* deadList );
    void fill_dead_TP( const char* file, DTDeadFlag* deadList );
    void fill_dead_RO( const char* file, DTDeadFlag* deadList );
    void fill_discCat( const char* file, DTDeadFlag* deadList );
    DTDeadFlag* dSum;
  };
}
