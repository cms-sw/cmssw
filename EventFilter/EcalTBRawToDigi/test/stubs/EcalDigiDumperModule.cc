/**
 * \file EcalDigiDumperModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2005/11/23 18:49:51 $
 * $Revision: 1.4 $
 * \author N. Amapane - S. Argiro'
 * \author G. Franzoni
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
  EcalDigiDumperModule(const edm::ParameterSet& ps){  
    verbosity= ps.getUntrackedParameter<int>("verbosity",1);
  }

  
 protected:
  int verbosity;

  void analyze( const edm::Event & e, const  edm::EventSetup& c){
    
    // retrieving crystal data from Event
    edm::Handle<EBDigiCollection>  digis;
    e.getByLabel("ecalEBunpacker", digis);

    // retrieving crystal PN diodes from Event
    edm::Handle<EcalPnDiodeDigiCollection>  PNs;
    e.getByLabel("ecalEBunpacker", PNs);

    
    cout << "\n\n ^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  digi cry collection size " << digis->size() << endl;
    cout << "                                  [EcalDigiDumperModule]  dumping some crystals\n"  << endl;
    short dumpConter =0;      

    if (verbosity>0)
      {
	for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	      ++digiItr ) {
	  
	  {
	    cout << "i-phi: " 
		 << (*digiItr).id().iphi() << " j-eta: " 
		 << (*digiItr).id().ieta()
		 << "   ";
	    for ( int i=0; i< (*digiItr).size() ; ++i ) {
	      if (i==5)	  cout << "\n\t";
	      cout <<  (*digiItr).sample(i) << " ";	  
	    }       
	    cout << " " << endl;
	    
	    if( (dumpConter++) > 10) break;
	    
	  } 
	}
      }
    
    

    cout << " \n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi PN collection.  Size: " << PNs->size() << endl;
    cout << "                                  [EcalDigiDumperModule]  dumping PN 1 "  << endl;
    if (verbosity>0)
      {
	for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
	  
	  cout << "PN num: " 
	       << (*pnItr).id().iPnId() << "\n";
	  
	  if ((*pnItr).id().iPnId() == 1){
	    for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {
	      cout <<  "sId: " << samId << " "
		   << (*pnItr).sample(samId) 
		   << "\t";
	    }// end loop on PN samples
	  }// end loop on PNs
	  
	}
      }
    
    
  } // produce

};// class EcalDigiDumperModule

DEFINE_FWK_MODULE(EcalDigiDumperModule)
  


