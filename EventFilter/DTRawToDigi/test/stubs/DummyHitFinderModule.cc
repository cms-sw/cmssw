 /**
 * \file DummyHitFinderModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2006/10/26 23:35:38 $
 * $Revision: 1.6 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <iostream>
#include <vector>



  class DummyHitFinderModule: public edm::EDAnalyzer{

  public:
    DummyHitFinderModule(const edm::ParameterSet& ps){}
    
  protected:

    void analyze( edm::Event const & e, const  edm::EventSetup& c){
      
    // ...Reconstruction, first step are the unpacking modules to
    // build digis...
      
      
      edm::Handle<DTDigiCollection> dtdigis;
      
      e.getByLabel("dtunpacker", dtdigis);
  

      DTDigiCollection::DigiRangeIterator detUnitIt;
      for (detUnitIt=dtdigis->begin();
	   detUnitIt!=dtdigis->end();
	   ++detUnitIt){
	
	for (DTDigiCollection::const_iterator digiIt = 
	       (*detUnitIt).second.first;
	     digiIt!=(*detUnitIt).second.second;
	     ++digiIt){
	  std::cout << "Digi: "  << *digiIt << std::endl;

	}// for cells
      }// for layers


    } // analyze
  };// class DummyHitFinderModule
 
DEFINE_FWK_MODULE(DummyHitFinderModule);



