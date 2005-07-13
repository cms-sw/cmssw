/**
 * \file DummyHitFinderModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2005/07/13 09:06:50 $
 * $Revision: 1.1 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <FWCore/Framework/interface/EDProducer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/DTDigis/interface/DTDigiCollection.h>
#include <DataFormats/DTDigis/interface/DTLayerId.h>
#include <iostream>
#include <vector>



  class DummyHitFinderModule: public edm::EDProducer{

  public:
    DummyHitFinderModule(const edm::ParameterSet& ps){}
    
  protected:

    void produce( edm::Event & e, const  edm::EventSetup& c){
      
    // ...Reconstruction, first step are the unpacking modules to
    // build digis...
      
      
      edm::Handle<DTDigiCollection> dtdigis;
      
      e.getByLabel("dtunpacker", dtdigis);
      // the vector of available layers
      std::vector<DTLayerId> layerids = dtdigis->layers();

      for (std::vector<DTLayerId>::const_iterator layer = layerids.begin();
	   layer!= layerids.end(); ++ layer){

	for (DTDigiCollection::const_iterator digi = 
	       dtdigis->layerBegin(*layer);
	     digi !=dtdigis->layerEnd(*layer); ++digi ){
	  
	  std::cout << "Layer: "  <<digi->layer() 
		    << " counts: "<<digi->countsTDC() << std::endl;

	}// for cells
      }// for layers


    } // produce
  };// class DummyHitFinderModule
 
DEFINE_FWK_MODULE(DummyHitFinderModule)



