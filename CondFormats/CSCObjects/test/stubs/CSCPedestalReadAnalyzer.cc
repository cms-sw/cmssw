/*----------------------------------------------------------------------

Toy EDProducers and EDProducts for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
#include "CondFormats/DataRecord/interface/CSCPedestalsRcd.h"

using namespace std;

namespace edmtest {
  class CSCPedestalReadAnalyzer : public edm::EDAnalyzer {
  public:
    explicit CSCPedestalReadAnalyzer(edm::ParameterSet const& p) {}
    explicit CSCPedestalReadAnalyzer(int i) {}
    ~CSCPedestalReadAnalyzer() override {}
    void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  private:
  };

  void CSCPedestalReadAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    // Context is not used.
    std::cout << " I AM IN RUN NUMBER " << e.id().run() << std::endl;
    std::cout << " ---EVENT NUMBER " << e.id().event() << std::endl;
    edm::ESHandle<CSCPedestals> pPeds;
    context.get<CSCPedestalsRcd>().get(pPeds);

    //call tracker code
    //
    /*
    int layerID=4;
    //CSCPedestals* myped=const_cast<CSCPedestals*>(pPeds.product());
    const CSCPedestals* myped=pPeds.product();
    std::map<int,std::vector<CSCPedestals::Item> >::const_iterator it=myped->pedestals.find(layerID);
    std::cout << "looking for CSC layer: " << layerID<<std::endl;
    if( it!=myped->pedestals.end() ){
      std::cout<<"layer id found "<<it->first<<std::endl;
      std::vector<CSCPedestals::Item>::const_iterator pedit;
      for( pedit=it->second.begin(); pedit!=it->second.end(); ++pedit ){
	std::cout << "  ped:  " <<pedit->ped << " rms: " << pedit->rms
		  << std::endl;
      }
    }
    */
    const CSCPedestals* myped = pPeds.product();
    std::map<int, std::vector<CSCPedestals::Item> >::const_iterator it;
    for (it = myped->pedestals.begin(); it != myped->pedestals.end(); ++it) {
      std::cout << "layer id found " << it->first << std::endl;
      std::vector<CSCPedestals::Item>::const_iterator pedit;
      for (pedit = it->second.begin(); pedit != it->second.end(); ++pedit) {
        std::cout << "  ped:  " << pedit->ped << " rms: " << pedit->rms << std::endl;
      }
    }
  }
  DEFINE_FWK_MODULE(CSCPedestalReadAnalyzer);
}  // namespace edmtest
