/**
 * \file DummyHitFinderModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2005/07/13 12:56:02 $
 * $Revision: 1.2 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/DTDigis/interface/DTDigiCollection.h>
#include <DataFormats/DTDigis/interface/DTLayerId.h>
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
	       detUnitIt->second.first;
	     digiIt!=detUnitIt->second.second;
	     ++digiIt){
	  std::cout << "Layer: "  <<digiIt->layer() 
		    << " counts: "<<digiIt->countsTDC() << std::endl;

	}// for cells
      }// for layers


    } // analyze
  };// class DummyHitFinderModule
 
DEFINE_FWK_MODULE(DummyHitFinderModule)



