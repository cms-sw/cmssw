/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
#include "CondFormats/DataRecord/interface/CSCDBPedestalsRcd.h"

using namespace std;

namespace edmtest
{
  class CSCPedestalDBReadAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  CSCPedestalDBReadAnalyzer(edm::ParameterSet const& p) 
    { }
    explicit  CSCPedestalDBReadAnalyzer(int i) 
    { }
    virtual ~ CSCPedestalDBReadAnalyzer() { }
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
  
  void
  CSCPedestalDBReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context)
  {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::ESHandle<CSCDBPedestals> pPeds;
    context.get<CSCDBPedestalsRcd>().get(pPeds);
    
    const CSCDBPedestals* myped=pPeds.product();
    std::vector<CSCDBPedestals::Item>::const_iterator it;
    
    for( it=myped->pedestals.begin();it!=myped->pedestals.end(); ++it ){
      //no global variables
	  //   counter++;
	  //DBPedestalFile<<it->ped<<"  "<<it->rms<<std::endl;
    }
  }
  DEFINE_FWK_MODULE(CSCPedestalDBReadAnalyzer);
}

