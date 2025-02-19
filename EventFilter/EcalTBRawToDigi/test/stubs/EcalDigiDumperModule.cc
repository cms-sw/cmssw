/**
 * \file EcalDigiDumperModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2008/01/22 18:59:17 $
 * $Revision: 1.20 $
 * \author N. Amapane - S. Argiro'
 * \author G. Franzoni
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>
#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>


#include <iostream>
#include <vector>


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
    listTowers     = ps.getUntrackedParameter<std::vector<int> >("listTowers", std::vector<int>());
    listPns        = ps.getUntrackedParameter<std::vector<int> >("listPns", std::vector<int>());
   
    
    inputIsOk = true;
    // consistency checks checks
    
    if ( (!(mode==1))  &&  (!(mode==2)) )
      {  
	std::cout << "[EcalDigiDumperModule] parameter mode set to: " << mode 
	     << ". Only mode 1 and 2 are allowed, returning." << std::endl;
	inputIsOk = false;
	return;
      }
    
    
    std::vector<int>::iterator intIter;
    
    for (intIter = listChannels.begin(); intIter != listChannels.end(); intIter++)
      {  
	if ( ((*intIter) < 1) ||  (1700 < (*intIter)) )       {  
	  std::cout << "[EcalDigiDumperModule] ic value: " << (*intIter) << " found in listChannels. "
	       << " Valid range is 1-1700. Returning." << std::endl;
	  inputIsOk = false;
	  return;
	}
      }
    
    
    for (intIter = listTowers.begin(); intIter != listTowers.end(); intIter++)
      {  
	if ( ((*intIter) < 1) ||  (70 < (*intIter)) )       {  
	  std::cout << "[EcalDigiDumperModule] ic value: " << (*intIter) << " found in listTowers. "
	       << " Valid range is 1-70. Returning." << std::endl;
	  inputIsOk = false;
	  return;
	}
      }
    
    for (intIter = listPns.begin(); intIter != listPns.end(); intIter++)
      {  
	if ( ((*intIter) < 1) ||  (10 < (*intIter)) )       {  
	  std::cout << "[EcalDigiDumperModule] pn number : " << (*intIter) << " found in listPns. "
	       << " Valid range is 1-10. Returning." << std::endl;
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
  std::vector<int> listTowers;
  std::vector<int> listPns;




  void analyze( const edm::Event & e, const  edm::EventSetup& c){

    if (!inputIsOk) return;
    
    
    // retrieving crystal data from Event
    edm::Handle<EBDigiCollection>  digis;
    e.getByLabel("ecalEBunpacker", "ebDigis", digis);

    // retrieving crystal PN diodes from Event
    edm::Handle<EcalPnDiodeDigiCollection>  PNs;
    e.getByLabel("ecalEBunpacker", PNs);

    // retrieve collection of errors in the mem gain data
    edm::Handle<EcalElectronicsIdCollection> gainMem;
    e.getByLabel("ecalEBunpacker", "EcalIntegrityMemChIdErrors", gainMem);
    
    // retrieve collection of errors in the mem gain data
    edm::Handle<EcalElectronicsIdCollection> MemId;
    e.getByLabel("ecalEBunpacker", "EcalIntegrityMemTtIdErrors", MemId);

    
    std::cout << "\n\n";

    if(gainMem->size() && memErrors) {  
      std::cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  Size of collection of mem gain errors is: " << gainMem->size() << std::endl;
      std::cout << "                                  [EcalDigiDumperModule]  dumping the bit gain errors\n"  << std::endl;
      for (EcalElectronicsIdCollection::const_iterator errItr= gainMem->begin();
	   errItr  != gainMem->end(); 
	   ++errItr ) {
	EcalElectronicsId  id = (*errItr);
	    std::cout << "channel: dccNum= " << id.dccId() 
		 << "\t tower= " << id.towerId() 
		 << "\t channelNum= " << id.channelId()
		 << " has problems in the gain bits" << std::endl;
      }// end of loop on gain errors in the mem
      }// end if
      

        
    if(MemId->size() && memErrors) {  
      std::cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  Size of collection of mem tt_block_id errors is: " << MemId->size() << std::endl;
      std::cout << "                                  [EcalDigiDumperModule]  dumping the mem tt_block_idb errors\n"  << std::endl;
      for (EcalElectronicsIdCollection::const_iterator errItr= MemId->begin();
	   errItr  != MemId->end(); 
	   ++errItr ) {
	EcalElectronicsId  id = (*errItr);
	    std::cout << "tower_block: dccNum= " << id.dccId() 
		 << "\t tower= " << id.towerId() 
		 << " has ID problems " << std::endl;
      }// end of loop tower_block_id errors in the mem
    }// end if
    

    short dumpCounter =0;      

    if (verbosity>0 && cryDigi && (mode==1) )
      {
	std::cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  digi cry collection size " << digis->size() << std::endl;
	std::cout << "                                  [EcalDigiDumperModule]  dumping first " << numChannel << " crystals\n";
	dumpCounter =0;      
	for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	      ++digiItr ) {
	  
	  {
	    if( (dumpCounter++) >= numChannel) break;
	    if (! ((EBDetId((*digiItr).id()).ism()==ieb_id) || (ieb_id==-1))  ) continue;
	    
		std::cout << "ic-cry: " 
		     << EBDetId((*digiItr).id()).ic() << " i-phi: " 
		     << EBDetId((*digiItr).id()).iphi() << " j-eta: " 
		     << EBDetId((*digiItr).id()).ieta();
		
		for ( unsigned int i=0; i< (*digiItr).size() ; ++i ) {
                  EBDataFrame df( *digiItr );
		  if (!(i%3)  )  std::cout << "\n\t";
		  std::cout << "sId: " << (i+1) << " "
		       <<  df.sample(i) << "\t";
		}       
		std::cout << " " << std::endl;

	  } 
	}
      }
    
    
    if (verbosity>0 && cryDigi && (mode==2) )
      {
	std::cout << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDumperModule]  digi cry collection size " << digis->size() << std::endl;
	for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	      ++digiItr ) {
	  {
	    int ic = EBDetId((*digiItr).id()).ic();
	    int tt = EBDetId((*digiItr).id()).tower().iTT();
	    
	    if (!  ((EBDetId((*digiItr).id()).ism()==ieb_id) || (ieb_id==-1))  ) continue;

	    std::vector<int>::iterator icIterCh;
	    std::vector<int>::iterator icIterTt;
	    icIterCh = find(listChannels.begin(), listChannels.end(), ic);
	    icIterTt = find(listTowers.begin(), listTowers.end(), tt);
	    if (icIterCh == listChannels.end()  
		&& icIterTt == listTowers.end() ) { continue; }

	    std::cout << "ic-cry: " 
		      << ic << " i-phi: " 
		      << EBDetId((*digiItr).id()).iphi() << " j-eta: " 
		      << EBDetId((*digiItr).id()).ieta() << " tower: "
		      << tt ;

	    for ( unsigned int i=0; i< (*digiItr).size() ; ++i ) {
              EBDataFrame df( *digiItr );
	      if (!(i%3)  )  std::cout << "\n\t";
	      std::cout << "sId: " << (i+1) << " " <<  df.sample(i) << "\t";
	    }       
	    std::cout << " " << std::endl;
	  } 
	}
      }
    
    


    if (verbosity>0 && pnDigi && (mode==1) )
      {
	std::cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi PN collection.  Size: " << PNs->size() << std::endl;
	std::cout << "                                  [EcalDigiDumperModule]  dumping first " << numPN << " PNs ";
	dumpCounter=0;
	for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
	  
	  if( (dumpCounter++) >= numPN) break;
	  if (! ((EcalPnDiodeDetId((*pnItr).id()).iDCCId()==ieb_id) || (ieb_id==-1))  ) continue;
	  
	  std::cout << "\nPN num: " << (*pnItr).id().iPnId();
	  
	  for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {
	    if (!(samId%3)  )  std::cout << "\n\t";
	    std::cout <<  "sId: " << (samId+1) << " "
		 << (*pnItr).sample(samId) 
		 << "\t";
	  }//  PN samples
	  
	}// PNs
      }
    
    

    if (verbosity>0 && pnDigi && (mode==2) )
      {
	std::cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi PN collection.  Size: " << PNs->size() << std::endl;

	for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
	  
	  int pnNum = (*pnItr).id().iPnId();

	  if (! ((EcalPnDiodeDetId((*pnItr).id()).iDCCId()==ieb_id) || (ieb_id==-1))  ) continue;

	  std::vector<int>::iterator pnIter;
	  pnIter = find(listPns.begin(), listPns.end(), pnNum);
	  if (pnIter == listPns.end()) { continue; }
	  
	  std::cout << "\nPN num: " << (*pnItr).id().iPnId();
	  for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {
	    if (!(samId%3)  )  std::cout << "\n\t";
	    std::cout <<  "sId: " << (samId+1) << " "
		 << (*pnItr).sample(samId) 
		 << "\t";
	  }//  PN samples
	  
	}// PNs
      }
    
    

//     // retrieving crystal TP from the Event
//     edm::Handle<EcalTrigPrimDigiCollection>  primitives;
//     e.getByLabel("ecalEBunpacker", primitives);
    
//     if (verbosity>0 && tpDigi)
//       {
// 	std::cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDumperModule  digi TP collection.  Size: " << primitives->size() << std::endl;
// 	std::cout << "                                  [EcalDigiDumperModule]  dumping primitives "  << std::endl;
// 	for ( EcalTrigPrimDigiCollection::const_iterator TPtr = primitives->begin();
// 	      ( TPtr != primitives->end()  && (TPtr-primitives->begin())<4 ); 
// 		++TPtr ) {

// 	  if (!  ((EcalTrigTowerDetId((*TPtr).id()).iDCC()==ieb_id) || (ieb_id==-1))   ) continue;

// 	  std::cout << "[EcalDigiDumperModule] tower: " << ( (TPtr-primitives->begin()) +1) 
// 	       << "\n" << (*TPtr) << std::endl;
// 	}
//       }


 
  } // produce

};// class EcalDigiDumperModule

DEFINE_FWK_MODULE(EcalDigiDumperModule);
