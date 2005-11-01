/**
 * \file EcalDigiDumperModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2005/08/05 14:34:03 $
 * $Revision: 1.2 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <iostream>
#include <vector>


using namespace cms;
using namespace std;


class EcalDigiDumperModule: public edm::EDAnalyzer{
  
public:
  EcalDigiDumperModule(const edm::ParameterSet& ps){   }
  
protected:
  
  void analyze( const edm::Event & e, const  edm::EventSetup& c){
    
    
    edm::Handle<EBDigiCollection>  digis;
    e.getByLabel("ecalEBunpacker", digis);
    
    cout << " ^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  collection size " << digis->size() << endl;
    
    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {
	
      cout << "\nDump samples for this event: i-phi: " 
	   << (*digiItr).id().iphi() << " j-eta: " 
	   << (*digiItr).id().ieta()
	   << "   ";
      for ( int i=0; i< (*digiItr).size() ; ++i ) {
	cout <<  (*digiItr).sample(i) << " ";
      }       
      cout << " " << endl;
      
      
    }	
    
    
    
    
    
    
  } // produce
};// class EcalDigiDumperModule

DEFINE_FWK_MODULE(EcalDigiDumperModule)
  


