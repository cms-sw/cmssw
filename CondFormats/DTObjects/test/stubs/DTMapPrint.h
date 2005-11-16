
/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"

using namespace std;

namespace edmtest {
  class DTMapPrint : public edm::EDAnalyzer
  {
  public:
    explicit  DTMapPrint(edm::ParameterSet const& p);
    explicit  DTMapPrint(int i) ;
    virtual ~ DTMapPrint();
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
}
