
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

//#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
//#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

using namespace std;

namespace edmtest {
  class DTMapWrite : public edm::EDAnalyzer
  {
  public:
    explicit  DTMapWrite(edm::ParameterSet const& p);
    explicit  DTMapWrite(int i) ;
    virtual ~ DTMapWrite();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
