/**
 * \file EcalDigiDisplay.cc 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2007/11/29 14:13:46 $
 * $Revision: 1.1 $
 * \author N. Amapane - S. Argiro'
 * \author G. Franzoni - Keti Kaadze
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "CaloOnlineTools/EcalTools/interface/EcalDigiDisplay.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"

#include <iostream>
#include <vector>
#include <set>
#include <map>

//==========================================================================
EcalDigiDisplay::EcalDigiDisplay(const edm::ParameterSet& ps) {
//=========================================================================
  
  ebDigiCollection_ = ps.getParameter<std::string>("ebDigiCollection");
  eeDigiCollection_ = ps.getParameter<std::string>("eeDigiCollection");
  digiProducer_     = ps.getParameter<std::string>("digiProducer");

  requestedFeds_ = ps.getUntrackedParameter<std::vector<int> >("requestedFeds");
  requestedEbs_  = ps.getUntrackedParameter<std::vector<std::string> >("requestedEbs");

  
  cryDigi        = ps.getUntrackedParameter<bool>("cryDigi");
  ttDigi        = ps.getUntrackedParameter<bool>("ttDigi");
  pnDigi         = ps.getUntrackedParameter<bool>("pnDigi");
  tpDigi         = ps.getUntrackedParameter<bool>("tpDigi");
  fedIsGiven     = ps.getUntrackedParameter<bool>("fedIsGiven");
  ebIsGiven      = ps.getUntrackedParameter<bool>("ebIsGiven");
  
  mode           = ps.getUntrackedParameter<int>("mode");
  numChannel     = ps.getUntrackedParameter<int>("numChannel");
  numPN          = ps.getUntrackedParameter<int>("numPN");
  
  listChannels   = ps.getUntrackedParameter<std::vector<int> >("listChannels");
  listTowers     = ps.getUntrackedParameter<std::vector<int> >("listTowers");
  listPns        = ps.getUntrackedParameter<std::vector<int> >("listPns");

  //Consistancy checks
  std::vector<int>::iterator fedIter;
  std::vector<int>::iterator intIter;
  bool inputIsOk = true;
  for (fedIter = requestedFeds_.begin(); fedIter!=requestedFeds_.end(); ++fedIter) {
    if ( (*fedIter) > 609 && (*fedIter) < 646) {                          // if EB SM is being picked up
      // Check with channels 
      for (intIter = listChannels.begin(); intIter != listChannels.end(); intIter++)  {  
	if ( ((*intIter) < 1) ||  (1700 < (*intIter)) )       {  
	  edm::LogError("EcalDigiDisplay") << "[EcalDigiDisplay] ic value: " << (*intIter) << " found in listChannels. "
						<< " Valid range is 1-1700. Returning.";
	  inputIsOk = false;
	  return;
	}
      }
      //Check with Towers
      if ( ttDigi ) {
	for (intIter = listTowers.begin(); intIter != listTowers.end(); intIter++) {
	  
	  if ( ((*intIter) < 1) ||  (70 < (*intIter)) )       {  
	    edm::LogError("EcalDigiDisplay") << "[EcalDigiDisplay] ic value: " << (*intIter) << " found in listTowers. "
						  << " Valid range for EB SM is 1-70. Returning.";
	    inputIsOk = false;
	    return;
	  }
	}
      }
    } else if ( ((*fedIter)>600&&(*fedIter)<610) || ((*fedIter)>645&&(*fedIter)<655) ) {              //if EE DCC is being picked up
      if (ttDigi) {
	//Check with Towers
	for (intIter = listTowers.begin(); intIter != listTowers.end(); intIter++) { 
	  if ( (*intIter) > 34 )       { 
	    edm::LogError("EcalDigiDisplay") << "[EcalDigiDisplay] ic value: " << (*intIter) << " found in listTowers. "
						  << " Valid range for EE DCC is 1-34. Returning.";
	    inputIsOk = false;
	    return;
	  }
	}
      }
    } else {
      edm::LogError("EcalDigiDisplay") << "[EcalDigiDisplay] FED id: "<<(*intIter)<<"found in listFeds."
					    << "Valid range is [601-654]. Returning. ";
      inputIsOk = false;
      return;
    }
  }
  //Check with Pns
  for (intIter = listPns.begin(); intIter != listPns.end(); intIter++) {
    if ( ((*intIter) < 1) ||  (10 < (*intIter)) )       {  
      edm::LogError("EcalDigiDisplay") << "[EcalDigiDisplay] pn number : " << (*intIter) << " found in listPns. "
					    << " Valid range is 1-10. Returning.";
      inputIsOk = false;
      return;
    }
  }

  fedMap = new EcalFedMap();
  //if EB/EE is given convert to FED id
  if ( !fedIsGiven ) {
    if ( ebIsGiven ) {
      requestedFeds_.clear();
      for (std::vector<std::string>::const_iterator ebItr = requestedEbs_.begin(); 
	                                            ebItr!= requestedEbs_.end();  ++ebItr) {
	requestedFeds_.push_back(fedMap->getFedFromSlice(*ebItr));
      }
    }
    fedIsGiven = true;
  }
}
//=========================================================================
EcalDigiDisplay::~EcalDigiDisplay() {
//=========================================================================
  //delete *;
}
    
//========================================================================
void EcalDigiDisplay::beginJob(const edm::EventSetup& c) {
//========================================================================
  edm::LogInfo("EcalDigiDisplay") << "entering beginJob! ";
}

//========================================================================
void EcalDigiDisplay::analyze( const edm::Event & e, const  edm::EventSetup& c) {
//========================================================================

  if (!inputIsOk) return;
  
  //Get DCC headers
  edm::Handle<EcalRawDataCollection> dccHeader;
  try {
    e.getByLabel(digiProducer_,dccHeader);
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "Can't get DCC Headers!";
  } 

  //
  bool ebDigisFound = false;
  bool eeDigisFound = false;
  bool pnDigisFound = false;
  // retrieving crystal data from Event
  edm::Handle<EBDigiCollection>  eb_digis;    
  try {
    e.getByLabel(digiProducer_,ebDigiCollection_, eb_digis);
    ebDigisFound = true;
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "EB Digis were not found!";
  }
  
  //
  edm::Handle<EEDigiCollection>  ee_digis;    
  try {
    e.getByLabel(digiProducer_,eeDigiCollection_, ee_digis);
    eeDigisFound = true;
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "EE Digis were not found!";
  }
  
  // retrieving crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection>  pn_digis;
  try {
    e.getByLabel(digiProducer_, pn_digis);
    pnDigisFound = true;
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "PNs were not found!";
  }
  
  if ( cryDigi ) {
    if ( ebDigisFound )
      readEBDigis(eb_digis, mode);
    if ( eeDigisFound )
      readEEDigis(ee_digis, mode);
    if ( !(ebDigisFound || eeDigisFound) ) {
      edm::LogWarning("EcalDigiUnpackerModule") << "No Digis were found! Returning..";
      return;
    }
  }
  
  if ( pnDigi ) {
    if (pnDigisFound )
      readPNDigis(pn_digis, mode);
  }
}

///////////////////////////////////
// FUNCTIONS
//////////////////////////////////

void EcalDigiDisplay::readEBDigis (edm::Handle<EBDigiCollection> digis, int Mode) {

  EcalElectronicsMapping* theMapping    = new EcalElectronicsMapping();
  int dumpDigiCounter = 0;
  
  for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	++digiItr ) {		
    //Make sure that digis are form right SM
    EBDetId detId = EBDetId((*digiItr).id());
    EcalElectronicsId elecId = theMapping->getElectronicsId(detId);

    int FEDid = elecId.dccId() + 600;
    std::vector<int>::iterator fedIter = find(requestedFeds_.begin(), requestedFeds_.end(), FEDid); 
    if (fedIter ==  requestedFeds_.end()) continue;
    
    //If we are here the digis are from given SM
    
    //Check if Mode is set 1 or 2 
     if ( Mode ==1 ) {
       edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDisplay]  digi cry collection size " << digis->size();
       edm::LogInfo("EcalDigiDisplay") << "                       [EcalDigiDisplay]  dumping first " << numChannel << " crystals\n";
      //It will break if all required digis are dumpped
      if( (dumpDigiCounter++) >= numChannel) break;     
    } else if  ( Mode==2 ) {
      int ic = EBDetId((*digiItr).id()).ic();
      int tt = EBDetId((*digiItr).id()).tower().iTT();
      
      std::vector<int>::iterator icIterCh;
      std::vector<int>::iterator icIterTt;
      icIterCh = find(listChannels.begin(), listChannels.end(), ic);
      icIterTt = find(listTowers.begin(), listTowers.end(), tt);
      if (icIterCh == listChannels.end() && icIterTt == listTowers.end() ) continue;   
      edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDisplay]  digi cry collection size " << digis->size();
    } else {
      edm::LogInfo("EcalDigiDisplay") << "[EcalDigiDisplay] parameter mode set to: " << Mode
					   << ". Only mode 1 and 2 are allowed. Returning...";
      inputIsOk = false;
      return;
    }
    
    std::cout << "FEDID: " << FEDid << std::endl;
    std::cout << "Tower: " << EBDetId((*digiItr).id()).tower().iTT()
              <<" ic-cry: " 
	      << EBDetId((*digiItr).id()).ic() << " i-phi: " 
	      << EBDetId((*digiItr).id()).iphi() << " j-eta: " 
	      << EBDetId((*digiItr).id()).ieta() << std::endl;
    //Get Samples
    for ( unsigned int i=0; i< (*digiItr).size() ; ++i ) {
      EBDataFrame df( *digiItr );
      if (!(i%3)  )  std::cout << "\n\t";
      std::cout << "sId: " << (i+1) << " " <<  df.sample(i) << "\t";
    }       
    std::cout << " " << std::endl;
  }
  delete theMapping;
}

//Function for EE Digis
void EcalDigiDisplay::readEEDigis (edm::Handle<EEDigiCollection> digis, int Mode) {

  //For Endcap so far works only  Mode 2
  if ( Mode!=2 ) {
    std::cout << "For Endcap mode needs to be set to 2" << std::endl;
    return;
  }
  
  EcalElectronicsMapping* theMapping   = new EcalElectronicsMapping(); 
  
  for ( EEDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	++digiItr ) {		
    
    //Make sure that digis are form requested place
    EEDetId detId = EEDetId((*digiItr).id());
    EcalElectronicsId elecId = theMapping->getElectronicsId(detId);

    int FEDid = elecId.dccId() + 600;
    std::vector<int>::iterator fedIter = find(requestedFeds_.begin(), requestedFeds_.end(), FEDid);
    if (fedIter ==  requestedFeds_.end()) continue;

    edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDisplay]  digi cry collection size " << digis->size();
    
    int crystalId = 10000 * FEDid + 100 * elecId.towerId() + 5 * (elecId.stripId()-1)+elecId.xtalId();
    int chId = elecId.towerId();    // this is a channel in Endcap DCC, sometimes also called as Super Crystal

    std::vector<int>::iterator icIterCh;
    std::vector<int>::iterator icIterTt;
    icIterCh = find(listChannels.begin(), listChannels.end(), crystalId);
    icIterTt = find(listTowers.begin(), listTowers.end(), chId);
    if ( icIterCh == listChannels.end() &&  icIterTt == listTowers.end() ) continue; 
    
    std::cout << "FEDID: " << FEDid << std::endl;
    std::cout << "Tower: " << elecId.towerId()    
	      << "crystalId: " 
	      << crystalId << " i-x: " 
	      << EEDetId((*digiItr).id()).ix() << " j-y: " 
	      << EEDetId((*digiItr).id()).iy() << std::endl;
    
    //Get samples 
    for ( unsigned int i=0; i< (*digiItr).size() ; ++i ) {
      EEDataFrame df( *digiItr );
      if (!(i%3)  )  std::cout << "\n\t";
      std::cout << "sId: " << (i+1) << " " <<  df.sample(i) << "\t";
    }       
    std::cout << " " << std::endl;
  }
  delete theMapping;
}

void EcalDigiDisplay::readPNDigis(edm::Handle<EcalPnDiodeDigiCollection> PNs, int Mode ) {

  int pnDigiCounter = 0;
  
  //Loop over PN digis
  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
    EcalPnDiodeDetId pnDetId = EcalPnDiodeDetId((*pnItr).id());
    //Make sure that we look at the requested place
    int FEDid = pnDetId.iDCCId() + 600;
    std::vector<int>::iterator fedIter = find(requestedFeds_.begin(), requestedFeds_.end(), FEDid);
    if (fedIter ==  requestedFeds_.end()) continue;
    int pnNum = (*pnItr).id().iPnId();
    
    //If Mode is 1
      if ( Mode == 1) {
	edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDisplay  digi PN collection.  Size: " << PNs->size();
	edm::LogInfo("EcalDigiDisplay") << "                       [EcalDigiDisplay]  dumping first " << numPN << " PNs ";
	
	if ( (pnDigiCounter++) >= numPN ) break;
      } else if ( Mode == 2) {
	edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDisplay  digi PN collection.  Size: " << PNs->size();
	
	// Check that we look at PN from the given list
	std::vector<int>::iterator pnIter;
	pnIter = find(listPns.begin(), listPns.end(), pnNum);
	if (pnIter == listPns.end())  continue; 
      } else {
	edm::LogError("EcalDigiDisplay")<< "[EcalDigiDisplay] parameter mode set to: " << Mode
					     << ". Only mode 1 and 2 are allowed. Returning...";
	inputIsOk = false;
	return;
      }
      
      std::cout << "DCCID: " << pnDetId.iDCCId() << std::endl;
      std::cout << "\nPN num: " << (*pnItr).id().iPnId();
      for ( int samId=0; samId < (*pnItr).size() ; samId++ ) {
	if (!(samId%3)  )  std::cout << "\n\t";
	std::cout <<  "sId: " << (samId+1) << " "
		  << (*pnItr).sample(samId) 
		  << "\t";
      }
  }
}

//===================================================
void EcalDigiDisplay::endJob() {
//==================================================
  edm::LogInfo("EcalDigiDisplay") << "DONE!.... " ;
}

 //     // retrieving crystal TP from the Event
 //     edm::Handle<EcalTrigPrimDigiCollection>  primitives;
 //     e.getByLabel("ecalEBunpacker", primitives);

 //     if (verbosity>0 && tpDigi)
 //       {
 // 	std::cout << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDisplay  digi TP collection.  Size: " << primitives->size() << std::endl;
 // 	std::cout << "                                  [EcalDigiDisplay]  dumping primitives "  << std::endl;
// 	for ( EcalTrigPrimDigiCollection::const_iterator TPtr = primitives->begin();
// 	      ( TPtr != primitives->end()  && (TPtr-primitives->begin())<4 ); 
// 		++TPtr ) {

// 	  if (!  ((EcalTrigTowerDetId((*TPtr).id()).iDCC()==ieb_id) || (ieb_id==-1))   ) continue;

// 	  std::cout << "[EcalDigiDisplay] tower: " << ( (TPtr-primitives->begin()) +1) 
// 	       << "\n" << (*TPtr) << std::endl;
// 	}
//       }


//} // produce

//};// class EcalDigiDisplay

DEFINE_FWK_MODULE(EcalDigiDisplay);

