
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

//#include <stdexcept>
//#include <string>
//#include <iostream>
//#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/ESHandle.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "CondFormats/DTObjects/interface/DTT0.h"
//#include "CondFormats/DataRecord/interface/DTT0Rcd.h"

using namespace std;

namespace edmtest {
  class DTT0Print : public edm::EDAnalyzer
  {
  public:
    explicit  DTT0Print(edm::ParameterSet const& p);
    explicit  DTT0Print(int i) ;
    virtual ~ DTT0Print();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
