/**
 * \file EcalDigiDumperModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2007/03/04 13:10:16 $
 * $Revision: 1.11 $
 * \author N. Amapane - S. Argiro'
 * \author G. Franzoni
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

#include <iostream>
#include <vector>


using namespace cms;
using namespace std;


class EcalDigiDumperModule: public edm::EDAnalyzer{
  
 public:
  EcalDigiDumperModule(const edm::ParameterSet& ps){  
    verbosity      = ps.getUntrackedParameter<int>("verbosity",1);

    ieb_id         = ps.getUntrackedParameter<int>("ieb_id",-1);

    memErrors      = ps.getUntrackedParameter<bool>("memErrors",true);
    cryDigi        = ps.getUntrackedParameter<bool>("cryDigi",true);
    pnDigi         = ps.getUntrackedParameter<bool>("pnDigi",true);
    tpDigi         = ps.getUntrackedParameter<bool>("tpDigi",true);


    mode           = ps.getUntrackedParameter<int>("mode",2);

    numChannel     = ps.getUntrackedParameter<int>("numChannel",10);
    numPN          = ps.getUntrackedParameter<int>("numPN",2);
    
    listChannels   = ps.getUntrackedParameter<std::vector<int> >("listChannels", std::vector<int>());
    listPns        = ps.getUntrackedParameter<std::vector<int> >("listPns", std::vector<int>());
   
    
    inputIsOk = true;
    // consistency checks checks
    
    if ( (!(mode==1))  &&  (!(mode==2)) )
      {  
	cout << "[EcalDigiDumperModule] parameter mode set to: " << mode 
	     << ". Only 1 and 2 are allowed, returning." << endl;
	inputIsOk = false;
	return;
      }
    
    
    vector<int>::iterator intIter;
    
    for (intIter = listChannels.begin(); intIter != listChannels.end(); intIter++)
      {  
	if ( ((*intIter) < 1) ||  (1700 < (*intIter)) )       {  
	  cout << "[EcalDigiDumperModule] ic value: " << (*intIter) << " found in listChannels. "
	       << " Valid range is 1-1700. Returning." << endl;
	  inputIsOk = false;
	  return;
	}
      }
    
    for (intIter = listPns.begin(); intIter != listPns.end(); intIter++)
      {  
	if ( ((*intIter) < 1) ||  (10 < (*intIter)) )       {  
	  cout << "[EcalDigiDumperModule] pn number : " << (*intIter) << " found in listPns. "
	       << " Valid range is 1-10. Returning." << endl;
	  inputIsOk = false;
	  return;
	}
      }
    
  }
  
  
 protected:
  int verbosity;
  bool memErrors;
  bool cryDigi;
  bool pnDigi;
  bool tpDigi;

  bool inputIsOk;

  int ieb_id;

  int mode;

  int numChannel;
  int numPN;

  std::vector<int> listChannels;
  std::vector<int> listPns;




  void analyze( const edm::Event & e, const  edm::EventSetup& c){

    if (!inputIsOk) return;
    
    
    // retrieving crystal data from Event
    edm::Handle<EBDigiCollection>  digis;
    e.getByLabel("ecalEBunpacker", digis);

    // retrieving crystal PN diodes from Event
    edm::Handle<EcalPnDiodeDigiCollection>  PNs;
    e.getByLabel("ecalEBunpacker", PNs);

    // retrieve collection of errors in the mem gain data
    edm::Handle<EcalElectronicsIdCollection> gainMem;
    e.getByLabel("ecalEBunpacker", "EcalIntegrityMemChIdErrors", gainMem);
    
    // retrieve collection of errors in the mem gain data
    edm::Handle<EcalElectronicsIdCollection> MemId;
    e.getByLabel("ecalEBunpacker", "EcalIntegrityMemTtIdErrors", MemId);

    
    cout << "\n\n";

    if(gainMem->size() && memErrors) {  
      cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  Size of collection of mem gain errors is: " << gainMem->size() << endl;
      cout << "                                  [EcalDigiDumperModule]  dumping the bit gain errors\n"  << endl;
      for (EcalElectronicsIdCollection::const_iterator errItr= gainMem->begin();
	   errItr  != gainMem->end(); 
	   ++errItr ) {
	EcalElectronicsId  id = (*errItr);
	    cout << "channel: dccNum= " << id.dccId() 
		 << "\t tower= " << id.towerId() 
		 << "\t channelNum= " << id.channelId()
		 << " has problems in the gain bits" << endl;
      }// end of loop on gain errors in the mem
      }// end if
      

        
    if(MemId->size() && memErrors) {  
      cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  Size of collection of mem tt_block_id errors is: " << MemId->size() << endl;
      cout << "                                  [EcalDigiDumperModule]  dumping the mem tt_block_idb errors\n"  << endl;
      for (EcalElectronicsIdCollection::const_iterator errItr= MemId->begin();
	   errItr  != MemId->end(); 
	   ++errItr ) {
	EcalElectronicsId  id = (*errItr);
	    cout << "tower_block: dccNum= " << id.dccId() 
		 << "\t tower= " << id.towerId() 
		 << " has ID problems " << endl;
      }// end of loop tower_block_id errors in the mem
    }// end if
    

    short dumpCounter =0;      

    if (verbosity>0 && cryDigi && (mode==1) )
      {
	cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  digi cry collection size " << digis->size() << endl;
	cout << "                                  [EcalDigiDumperModule]  dumping first " << numChannel << " crystals\n";
	dumpCounter =0;      
	for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	      ++digiItr ) {
	  
	  {
	    if( (dumpCounter++) >= numChannel) break;
	    if (! ((EBDetId((*digiItr).id()).ism()==ieb_id) || (ieb_id==-1))  ) continue;
	    
		cout << "ic-cry: " 
		     << EBDetId((*digiItr).id()).ic() << " i-phi: " 
		     << EBDetId((*digiItr).id()).iphi() << " j-eta: " 
		     << EBDetId((*digiItr).id()).ieta();
		
		for ( int i=0; i< (*digiItr).size() ; ++i ) {
		  if (!(i%5)  )  cout << "\n\t";
		  cout << "sId: " << (i+1) << " "
		       <<  (*digiItr).sample(i) << "\t";
		}       
		cout << " " << endl;

	  } 
	}
      }
    
    
    if (verbosity>0 && cryDigi && (mode==2) )
      {
	cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  digi cry collection size " << digis->size() << endl;
	for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	      ++digiItr ) {
	  {
	    int ic = EBDetId((*digiItr).id()).ic();
	    
	    if (!  ((EBDetId((*digiItr).id()).ism()==ieb_id) || (ieb_id==-1))  ) continue;

	    vector<int>::iterator icIter;
	    icIter = find(listChannels.begin(), listChannels.end(), ic);
	    if (icIter == listChannels.end()) { continue; }
	    
	    cout << "ic-cry: " 
		 << EBDetId((*digiItr).id()).ic() << " i-phi: " 
		 << EBDetId((*digiItr).id()).iphi() << " j-eta: " 
		 << EBDetId((*digiItr).id()).ieta();
	    
	    for ( int i=0; i< (*digiItr).size() ; ++i ) {
	      if (!(i%5)  )  cout << "\n\t";
	      cout << "sId: " << (i+1) << " " <<  (*digiItr).sample(i) << "\t";
	    }       
	    cout << " " << endl;
	  } 
	}
      }
    
    


    if (verbosity>0 && pnDigi && (mode==1) )
      {
	cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi PN collection.  Size: " << PNs->size() << endl;
	cout << "                                  [EcalDigiDumperModule]  dumping first " << numPN << " PNs ";
	dumpCounter=0;
	for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
	  
	  if( (dumpCounter++) >= numPN) break;
	  if (! ((EcalPnDiodeDetId((*pnItr).id()).iDCCId()==ieb_id) || (ieb_id==-1))  ) continue;
	  
	  cout << "\nPN num: " << (*pnItr).id().iPnId();
	  
	  for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {
	    if (!(samId%5)  )  cout << "\n\t";
	    cout <<  "sId: " << (samId+1) << " "
		 << (*pnItr).sample(samId) 
		 << "\t";
	  }//  PN samples
	  
	}// PNs
      }
    
    

    if (verbosity>0 && pnDigi && (mode==2) )
      {
	cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi PN collection.  Size: " << PNs->size() << endl;

	for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
	  
	  int pnNum = (*pnItr).id().iPnId();

	  if (! ((EcalPnDiodeDetId((*pnItr).id()).iDCCId()==ieb_id) || (ieb_id==-1))  ) continue;

	  vector<int>::iterator pnIter;
	  pnIter = find(listPns.begin(), listPns.end(), pnNum);
	  if (pnIter == listPns.end()) { continue; }
	  
	  cout << "\nPN num: " << (*pnItr).id().iPnId();
	  for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {
	    if (!(samId%5)  )  cout << "\n\t";
	    cout <<  "sId: " << (samId+1) << " "
		 << (*pnItr).sample(samId) 
		 << "\t";
	  }//  PN samples
	  
	}// PNs
      }
    
    

    // retrieving crystal TP from the Event
    edm::Handle<EcalTrigPrimDigiCollection>  primitives;
    e.getByLabel("ecalEBunpacker", primitives);
    
    if (verbosity>0 && tpDigi)
      {
	cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi TP collection.  Size: " << primitives->size() << endl;
	cout << "                                  [EcalDigiDumperModule]  dumping primitives "  << endl;
	for ( EcalTrigPrimDigiCollection::const_iterator TPtr = primitives->begin();
	      ( TPtr != primitives->end()  && (TPtr-primitives->begin())<4 ); 
		++TPtr ) {

	  if (!  ((EcalTrigTowerDetId((*TPtr).id()).iDCC()==ieb_id) || (ieb_id==-1))   ) continue;

	  cout << "[EcalDigiDumperModule] tower: " << ( (TPtr-primitives->begin()) +1) 
	       << "\n" << (*TPtr) << endl;
	}
      }


 
  } // produce

};// class EcalDigiDumperModule

DEFINE_FWK_MODULE(EcalDigiDumperModule);
