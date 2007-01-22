
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

//#include "CondFormats/DTObjects/interface/DTTtrig.h"
//#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

using namespace std;

namespace edmtest {
  class DTTtrigWrite : public edm::EDAnalyzer
  {
  public:
    explicit  DTTtrigWrite(edm::ParameterSet const& p);
    explicit  DTTtrigWrite(int i) ;
    virtual ~ DTTtrigWrite();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
