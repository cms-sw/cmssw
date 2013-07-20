/**
 * \file EcalDigiDisplay.cc 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2011/10/10 09:05:21 $
 * $Revision: 1.4 $
 * \author Keti Kaadze
 * \author G. Franzoni
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "CaloOnlineTools/EcalTools/plugins/EcalDigiDisplay.h"
#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

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

  cryDigi       = ps.getUntrackedParameter<bool>("cryDigi");
  ttDigi        = ps.getUntrackedParameter<bool>("ttDigi");
  pnDigi      = ps.getUntrackedParameter<bool>("pnDigi");
   
  mode           = ps.getUntrackedParameter<int>("mode");
  listChannels   = ps.getUntrackedParameter<std::vector<int> >("listChannels");
  listTowers     = ps.getUntrackedParameter<std::vector<int> >("listTowers");
  listPns        = ps.getUntrackedParameter<std::vector<int> >("listPns");

  std::vector<int> listDefaults;
  listDefaults.push_back(-1);
  requestedFeds_ = ps.getUntrackedParameter<std::vector<int> >("requestedFeds",listDefaults);
  bool fedIsGiven = false;
  
  std::vector<std::string> ebDefaults;
  ebDefaults.push_back("none");
  requestedEbs_  = ps.getUntrackedParameter<std::vector<std::string> >("requestedEbs",ebDefaults);
  // FEDs and EBs
  if ( requestedFeds_[0] != -1 ) {
    edm::LogInfo("EcalDigiDisplay") << "FED id is given! Goining to beginRun! ";
    fedIsGiven = true;
  }else {
    if ( requestedEbs_[0] !="none" ) {
      //EB id is given and convert to FED id
      requestedFeds_.clear();
      fedMap = new EcalFedMap();
      for (std::vector<std::string>::const_iterator ebItr = requestedEbs_.begin(); 
	   ebItr!= requestedEbs_.end();  ++ebItr) {
	requestedFeds_.push_back(fedMap->getFedFromSlice(*ebItr));
      }
      delete fedMap;
    } else {
      //Select all FEDs in the Event
      for ( int i=601; i<655; ++i){
	requestedFeds_.push_back(i);
      }
    }
  }

  //Channel list
  listChannels = ps.getUntrackedParameter<std::vector<int> >("listChannels",listDefaults);
  //Tower list
  listTowers     = ps.getUntrackedParameter<std::vector<int> >("listTowers", listDefaults);
  
  //Consistancy checks:
  std::vector<int>::iterator fedIter;
  std::vector<int>::iterator intIter;
  inputIsOk = true;

  if ( fedIsGiven ) {
    for ( fedIter = requestedFeds_.begin(); fedIter!=requestedFeds_.end(); ++fedIter) {  
      if ( (*fedIter) < 601 || (*fedIter) > 655 ) {
	edm::LogError("EcalDigiDisplay") << " FED value: " << (*fedIter) << " found in requetsedFeds. "
					 << " Valid range is 601-654. Returning.";
	inputIsOk = false;
	return;
      }//Loop over requetsed FEDS
    } 
  }
  bool barrelSM  = false;  
  //Loop over and Check if Barrel SM is picked up
  for (fedIter = requestedFeds_.begin(); fedIter!=requestedFeds_.end(); ++fedIter) {
    if ( (*fedIter) > 609 && (*fedIter) < 646 && inputIsOk )      // if EB SM is being picked up
      barrelSM = true;
  }
  
  if ( barrelSM ) {
    if ( cryDigi ) {
    // Check with channels in Barrel
      for (intIter = listChannels.begin(); intIter != listChannels.end(); intIter++)  {  
	if ( ((*intIter) < 1) ||  (1700 < (*intIter)) )       {  
	  edm::LogError("EcalDigiDisplay") << " ic value: " << (*intIter) << " found in listChannels. "
					   << " Valid range is 1-1700. Returning.";
	  inputIsOk = false;
	  return;
	}
      }
    }
    //Check with Towers in Barrel
    if ( ttDigi ) {
      for (intIter = listTowers.begin(); intIter != listTowers.end(); intIter++) {
	
	if ( ((*intIter) < 1) ||  (70 < (*intIter)) )       {  
	  edm::LogError("EcalDigiDisplay") << " TT value: " << (*intIter) << " found in listTowers. "
					   << " Valid range for EB SM is 1-70. Returning.";
	  inputIsOk = false;
	  return;
	}
      }
    }
  }else  //if EE DCC is being picked up  
    if (ttDigi) {
      //Check with Towers in Endcap
      for (intIter = listTowers.begin(); intIter != listTowers.end(); intIter++) { 
	if ( (*intIter) > 34 )       { 
	  edm::LogError("EcalDigiDisplay") << " TT value: " << (*intIter) << " found in listTowers. "
					   << " Valid range for EE DCC is 1-34. Returning.";
	  inputIsOk = false;
	  return;
	}
      }
    }

  //PNs
  listPns     = ps.getUntrackedParameter<std::vector<int> >("listPns",listDefaults);
  /*
  if ( listPns[0] != -1 ) pnDigi = true;
  else {
    listPns.clear();
    for ( int i=1; i < 11; ++i ) {
      listPns.push_back(i);
    }
  }
  */
  if ( pnDigi ) {
    for (intIter = listPns.begin(); intIter != listPns.end(); intIter++) {
      if ( ((*intIter) < 1) ||  (10 < (*intIter)) )       {  
	edm::LogError("EcalDigiDisplay") << " Pn number : " << (*intIter) << " found in listPns. "
					 << " Valid range is 1-10. Returning.";
	inputIsOk = false;
	return;
      }
    }
  }
}
//=========================================================================
EcalDigiDisplay::~EcalDigiDisplay() {
//=========================================================================
  //delete *;
}
    
//========================================================================
void EcalDigiDisplay::beginRun(edm::Run const &, edm::EventSetup const& c) {
//========================================================================
  edm::LogInfo("EcalDigiDisplay") << "entering beginRun! ";

  edm::ESHandle<EcalElectronicsMapping> elecHandle;
    c.get<EcalMappingRcd>().get(elecHandle);
  ecalElectronicsMap_ = elecHandle.product();
}

//========================================================================
void EcalDigiDisplay::analyze( edm::Event const & e, edm::EventSetup const & c) {
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
    if ( eb_digis->size() != 0 )
      ebDigisFound = true;
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "EB Digis were not found!";
  }
  
  //
  edm::Handle<EEDigiCollection>  ee_digis;    
  try {
    e.getByLabel(digiProducer_,eeDigiCollection_, ee_digis);
    if ( ee_digis->size() != 0 )
      eeDigisFound = true;
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "EE Digis were not found!";
  }
  
  // retrieving crystal PN diodes from Event
  edm::Handle<EcalPnDiodeDigiCollection>  pn_digis;
  try {
    e.getByLabel(digiProducer_, pn_digis);
    if ( pn_digis->size() != 0)
      pnDigisFound = true;
  } catch (cms::Exception& ex) {
    edm::LogError("EcalDigiUnpackerModule") << "PNs were not found!";
  }

  //=============================
  //Call for funcitons
  //=============================
  if ( cryDigi || ttDigi ) {
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

  for ( EBDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	++digiItr ) {		

    EBDetId detId = EBDetId((*digiItr).id());
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);

    int FEDid = elecId.dccId() + 600;
    std::vector<int>::iterator fedIter = find(requestedFeds_.begin(), requestedFeds_.end(), FEDid); 
    if (fedIter ==  requestedFeds_.end()) continue;

    int ic = EBDetId((*digiItr).id()).ic();
    int tt = EBDetId((*digiItr).id()).tower().iTT();

    //Check if Mode is set 1 or 2 
    if ( Mode ==1 ) {
      edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ [EcalDigiDisplay]  digi cry collection size " << digis->size();
      edm::LogInfo("EcalDigiDisplay") << "                       [EcalDigiDisplay]  dumping first " << listChannels[0] << " crystals\n";
      //It will break if all required digis are dumpped
      if( ic  > listChannels[0]) continue;  
    } else if  ( Mode==2 ) {

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
}

//Function for EE Digis
void EcalDigiDisplay::readEEDigis (edm::Handle<EEDigiCollection> digis, int Mode) {

  //For Endcap so far works only  Mode 2
  if ( Mode!=2 ) {
    std::cout << "For Endcap mode needs to be set to 2" << std::endl;
    return;
  }
  
  for ( EEDigiCollection::const_iterator digiItr= digis->begin();digiItr != digis->end(); 
	++digiItr ) {		
    
    //Make sure that digis are form requested place
    EEDetId detId = EEDetId((*digiItr).id());
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(detId);

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
}

void EcalDigiDisplay::readPNDigis(edm::Handle<EcalPnDiodeDigiCollection> PNs, int Mode) {

  int pnDigiCounter = 0;

  //Loop over PN digis
  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {
    EcalPnDiodeDetId pnDetId = EcalPnDiodeDetId((*pnItr).id());
    //Make sure that we look at the requested place
    int FEDid = pnDetId.iDCCId() + 600;
    std::vector<int>::iterator fedIter = find(requestedFeds_.begin(), requestedFeds_.end(), FEDid);
    if (fedIter ==  requestedFeds_.end()) continue;
    int pnNum = (*pnItr).id().iPnId();
    
    if ( Mode == 1) {
      edm::LogInfo("EcalDigiDisplay") << "\n\n^^^^^^^^^^^^^^^^^^ EcalDigiDisplay  digi PN collection.  Size: " << PNs->size();
      edm::LogInfo("EcalDigiDisplay") << "                       [EcalDigiDisplay]  dumping first " << listPns[0] << " PNs ";
      
      if ( (pnDigiCounter++) >= listPns[0] ) break;
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


