// -*- C++ -*-
//
// Package:    SiStripMonitorPedestals
// Class:      SiStripMonitorPedestals
// 
/**\class SiStripMonitorDigi SiStripMonitorDigi.cc DQM/SiStripMonitorDigi/src/SiStripMonitorDigi.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Simone Gennai and Suchandra Dutta
//         Created:  Sat Feb  4 20:49:10 CET 2006
// $Id: SiStripMonitorPedestals.cc,v 1.11 2006/07/10 07:50:18 gennai Exp $
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

//#include "CalibFormats/SiStripObjects/interface/SiStripStructure.h" // these two will go away
//#include "CalibTracker/Records/interface/SiStripStructureRcd.h"     // these two will go away

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorPedestals.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/Framework/interface/Handle.h"
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// data formats

// std
#include <cstdlib>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

SiStripMonitorPedestals::SiStripMonitorPedestals(const edm::ParameterSet& iConfig)
{
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
  pedsPSet_ = conf_.getParameter<edm::ParameterSet>("PedestalsPSet");
  
  analyzed = false;
  signalCutPeds_ = 4;
  nEvTot_=0;
  apvFactory_=0;
  nIteration_=0;

  outPutFileName = conf_.getParameter<string>("OutPutFileName");

  theEventInitNumber_ =  pedsPSet_.getParameter<int>("NumberOfEventsForInit");
  theEventIterNumber_ = pedsPSet_.getParameter<int>("NumberOfEventsForIteration");
  NumCMstripsInGroup_ = pedsPSet_.getParameter<int>("NumCMstripsInGroup");
}


SiStripMonitorPedestals::~SiStripMonitorPedestals()
{
  if (apvFactory_) {delete apvFactory_;} 
}


void SiStripMonitorPedestals::beginJob(const edm::EventSetup& es){
   // retrieve parameters from configuration file
  using namespace edm;
 vector<uint32_t> SelectedDetIds;
 SelectedDetIds.clear();

 //ApvAnalysisFactory
 apvFactory_ = new ApvAnalysisFactory(pedsPSet_);

//getting det id from the fed cabling
  edm::ESHandle<SiStripFedCabling> cabling;
  es.get<SiStripFedCablingRcd>().get( cabling );
  fedCabling_ = const_cast<SiStripFedCabling*>( cabling.product() );

  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  // create SiStripFolderOrganizer
  SiStripFolderOrganizer folder_organizer;
  
  map<uint32_t, int> detIdApv;
  detIdApv.clear();
  //To do so I need to access cabling from FED and so on ... see rob's code
   // Retrieve FED ids from cabling map and iterate through 
  const vector<uint16_t>& fed_ids = fedCabling_->feds(); 
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) {  
    for ( uint16_t channel = 0; channel < 96; channel++ ) {
      const FedChannelConnection& conn = fedCabling_->connection( *ifed, channel );
      uint32_t key_id =  conn.detId();
      int napvs = (conn.nApvPairs())*2;
      if(key_id > 0 && napvs >0) { 
	
	//	int nAPVS = detId_pos->second;
	bool newDetId =   apvFactory_->instantiateApvs(key_id,napvs);
	if(newDetId ) {
	  ModMEs local_modmes;
	  SelectedDetIds.push_back(key_id);
	  string hid;
	  // set appropriate folder using SiStripFolderOrganizer
	  folder_organizer.setDetectorFolder(key_id); // pass the detid to this method
	  int nStrip  = napvs*128;
	  //Pedestals histos
	  hid = hidmanager.createHistoId("PedsPerStrip","det", key_id);
	  local_modmes.PedsPerStrip = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5); //to modify the size binning 

	  hid = hidmanager.createHistoId("PedsDistribution","det", key_id);
	  local_modmes.PedsDistribution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 300, 200, 500); //to modify the size binning 

	  hid = hidmanager.createHistoId("PedsEvolution","det", key_id);
	  local_modmes.PedsEvolution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 50, 0., 50.); //to modify the size binning 


	  //Noise histos
	  hid = hidmanager.createHistoId("CMSubNoisePerStrip","det", key_id);
	  local_modmes.CMSubNoisePerStrip = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5);

	  hid = hidmanager.createHistoId("RawNoisePerStrip","det", key_id);
	  local_modmes.RawNoisePerStrip = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5);

	  hid = hidmanager.createHistoId("CMSubNoiseProfile","det", key_id);
	  local_modmes.CMSubNoiseProfile = dbe_->bookProfile(hid, hid, nStrip,0.5,nStrip+0.5, 100, 0., 100.);

	  hid = hidmanager.createHistoId("RawNoiseProfile","det", key_id);
	  local_modmes.RawNoiseProfile = dbe_->bookProfile(hid, hid, nStrip,0.5,nStrip+0.5, 100, 0., 100.);


	  hid = hidmanager.createHistoId("NoisyStrips","det", key_id);
	  local_modmes.NoisyStrips = dbe_->book2D(hid, hid, nStrip,0.5,nStrip+0.5,6,-0.5,5.5);

	  hid = hidmanager.createHistoId("NoisyStripDistribution","det", key_id);
      	  local_modmes.NoisyStripDistribution = dbe_->book1D(hid, hid, 11, -0.5,10.5);

	  //Common Mode histos
	  hid = hidmanager.createHistoId("CMDistribution","det", key_id);
	  local_modmes.CMDistribution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 150, -15., 15.); 
    
	  // append to DigiMEs
	  DigiMEs.insert( std::make_pair(key_id, local_modmes));
	  //
	}
      }
    }
    
  }
  cout <<"Number of DETS "<<SelectedDetIds.size()<<endl;
}


// ------------ method called to produce the data  ------------
void SiStripMonitorPedestals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  //Increment # of Events
  nEvTot_++;
  using namespace edm;
  vector<uint32_t> SelectedDetIds;
  SelectedDetIds.clear(); 
  // retrieve producer name of input StripDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
    // get DigiCollection object from Event
    edm::Handle< edm::DetSetVector<SiStripRawDigi> > digi_collection;
    std::string digiType = "VirginRaw";
    //you have a collection as there are all the digis for the event for every detector
    iEvent.getByLabel(digiProducer, digiType, digi_collection);
    // loop over all MEs

    //Increase the number of iterations ...
    if((nEvTot_ - theEventInitNumber_)%theEventIterNumber_ == 1) nIteration_++;

    for (map<uint32_t, ModMEs >::const_iterator i = DigiMEs.begin() ; i!=DigiMEs.end() ; i++) {
      uint32_t detid = i->first; ModMEs local_modmes = i->second;
      // get iterators for digis belonging to one DetId, it is an iterator, i.e. one element of the vector
      
      vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = digi_collection->find( detid );
      if ( digis->data.empty() ) { 
	edm::LogError("MonitorDigi_tmp") << "[SiStripRawDigiToRaw::createFedBuffers] Zero digis found!"; 
      } 
      uint32_t id  = detid;
      //    cout <<"Data size "<<digis->data.size()<<endl;
      apvFactory_->update(id, (*digis));
      
      if(nEvTot_ > theEventInitNumber_) {
	if(local_modmes.CMDistribution != NULL){ 
	  vector<float> tmp;
	  tmp.clear();
	  apvFactory_->getCommonMode(id, tmp);
	  //unpacking the info looking at the right topology
	  int numberCMBlocks = int(128. / NumCMstripsInGroup_);
	  int ibin=0;
	  for (vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
	    int iapv = int (ibin/numberCMBlocks);
	    (local_modmes.CMDistribution)->Fill(iapv,static_cast<float>(*iped));
	    ibin++;
	    
	  }
	}
      }
      
      //asking for the status
      if((nEvTot_ - theEventInitNumber_)%theEventIterNumber_ == 1)
	{
	      
	  vector<float> tmp;
	  tmp.clear();
	  apvFactory_->getPedestal(id, tmp);
	  if(local_modmes.PedsPerStrip != NULL){ 
	    int numberOfApvs = int(tmp.size()/128.);
	    for(int i=0; i<numberOfApvs;i++){
	      vector<float> myPedPerApv;
	      apvFactory_->getPedestal(id, i, myPedPerApv);
	      float avarage = 0;
	      avarage = accumulate(myPedPerApv.begin(), myPedPerApv.end(), avarage);
	      avarage = avarage/128.;
	      (local_modmes.PedsEvolution)->setBinContent(i+1,nIteration_,avarage);
	      
	    }
	    int ibin=0;
	  
	    for (vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
	      int napv = int(ibin / 128.);
	      ibin++;
	      float last_value = (local_modmes.PedsPerStrip)->getBinContent(ibin);
	      if(last_value != 0.){
		(local_modmes.PedsPerStrip)->setBinContent(ibin,(static_cast<float>(*iped) + last_value)/2.);
	      }else{
		(local_modmes.PedsPerStrip)->setBinContent(ibin,static_cast<float>(*iped));
	      }
	      (local_modmes.PedsDistribution)->Fill(napv,static_cast<float>(*iped));
	    }
	  }
	  
	  if(local_modmes.CMSubNoisePerStrip != NULL && local_modmes.CMSubNoiseProfile != NULL){ 
	    tmp.clear();
	    apvFactory_->getNoise(id, tmp);
	    int ibin=0;
	    for (vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
	      ibin++;
		(local_modmes.CMSubNoiseProfile)->Fill(static_cast<double>(ibin*1.),static_cast<float>(*iped));

	      float last_value = (local_modmes.CMSubNoisePerStrip)->getBinContent(ibin);
	      if(last_value != 0.){
		(local_modmes.CMSubNoisePerStrip)->setBinContent(ibin,(static_cast<float>(*iped)+last_value)/2.);
	      }else{
		(local_modmes.CMSubNoisePerStrip)->setBinContent(ibin,static_cast<float>(*iped));
	      }
	    }
	  }

	  
	  if(local_modmes.RawNoisePerStrip != NULL && local_modmes.RawNoiseProfile != NULL){ 
	    tmp.clear();
	    apvFactory_->getRawNoise(id, tmp);
	    int ibin=0;
	    for (vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
	      ibin++;
	      (local_modmes.RawNoiseProfile)->Fill(static_cast<double>(ibin*1.),static_cast<float>(*iped));
	      float last_value = (local_modmes.RawNoisePerStrip)->getBinContent(ibin);
	      if(last_value != 0.){
		(local_modmes.RawNoisePerStrip)->setBinContent(ibin,(static_cast<float>(*iped)+last_value)/2.);
	      }else{
		(local_modmes.RawNoisePerStrip)->setBinContent(ibin,static_cast<float>(*iped));
	      }
	    }
	  }

	  if(local_modmes.NoisyStrips != NULL){ 
	    TkApvMask::MaskType temp;
	    apvFactory_->getMask(id, temp);
	    int ibin=0;
	    for (TkApvMask::MaskType::const_iterator iped=temp.begin(); iped!=temp.end();iped++) {
	      ibin++;
	      
	      if(nIteration_ <2){
		if(*iped == 1)
		  (local_modmes.NoisyStrips)->Fill(ibin,3.);
		if(*iped == 2)
		  (local_modmes.NoisyStrips)->Fill(ibin,4.);
		if(*iped == 0)
		  (local_modmes.NoisyStrips)->Fill(ibin,0.);
	      }else{
		(local_modmes.NoisyStrips)->Fill(ibin,static_cast<float>(*iped));
		(local_modmes.NoisyStripDistribution)->Fill(static_cast<float>(*iped));
	      }
	    }
	  }


	}
    }
  usleep(500000);
}
    


void SiStripMonitorPedestals::endJob(void){

      //  dbe_->showDirStructure();
//  dbe_->save(outPutFileName);
}

