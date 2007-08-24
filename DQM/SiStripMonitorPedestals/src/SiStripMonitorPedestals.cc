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
// $Id: SiStripMonitorPedestals.cc,v 1.23 2007/07/25 15:41:57 dutta Exp $
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

// std
#include <cstdlib>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>

SiStripMonitorPedestals::SiStripMonitorPedestals(const edm::ParameterSet& iConfig):
  dbe_(edm::Service<DaqMonitorBEInterface>().operator->()),
  conf_(iConfig),
  pedsPSet_(iConfig.getParameter<edm::ParameterSet>("PedestalsPSet")),
  analyzed(false),
  firstEvent(true),
  signalCutPeds_(4),
  nEvTot_(0),
  nIteration_(0),
  apvFactory_(0),
  outPutFileName(iConfig.getParameter<std::string>("OutPutFileName"))
{
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

 std::vector<uint32_t> SelectedDetIds;
 SelectedDetIds.clear();

 //ApvAnalysisFactory
 apvFactory_ = new ApvAnalysisFactory(pedsPSet_);

  //getting det id from the det cabling    
  es.get<SiStripDetCablingRcd>().get( detcabling );

  detcabling->addActiveDetectorsRawIds(SelectedDetIds);

  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  // create SiStripFolderOrganizer
  SiStripFolderOrganizer folder_organizer;

  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){
    
    uint32_t detid = *idetid;
    if ((detid) == 0){
      //FIXME insert a comment 
      continue;
    }

    int napvs = detcabling->nApvPairs(detid) * 2;
    int nStrip  = napvs*128;
    
    bool newDetId =   apvFactory_->instantiateApvs(detid,napvs);  
  
    if( newDetId ) {
      ModMEs local_modmes;
      std::string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(detid); // pass the detid to this method
      
      //Pedestals histos
      hid = hidmanager.createHistoId("PedsPerStrip","det", detid);
      local_modmes.PedsPerStrip = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5); //to modify the size binning 
      dbe_->tag(local_modmes.PedsPerStrip, detid);
      (local_modmes.PedsPerStrip)->setAxisTitle("Pedestal (ADC)  vs Strip Number ",1);

      hid = hidmanager.createHistoId("PedsDistribution","det", detid);
      local_modmes.PedsDistribution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 300, 200, 500); //to modify the size binning 
      dbe_->tag(local_modmes.PedsDistribution, detid);
      (local_modmes.PedsDistribution)->setAxisTitle("Apv Number",1);
      (local_modmes.PedsDistribution)->setAxisTitle("Mean Pedestal Value (ADC)",2);

      hid = hidmanager.createHistoId("PedsEvolution","det", detid);
      local_modmes.PedsEvolution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 50, 0., 50.); //to modify the size binning 
      dbe_->tag(local_modmes.PedsEvolution, detid);
      (local_modmes.PedsEvolution)->setAxisTitle("Apv Number",1);
      (local_modmes.PedsEvolution)->setAxisTitle("Iteration Number",2);

      //Noise histos
      hid = hidmanager.createHistoId("CMSubNoisePerStrip","det", detid);
      local_modmes.CMSubNoisePerStrip = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5);
      dbe_->tag(local_modmes.CMSubNoisePerStrip, detid);
      (local_modmes.CMSubNoisePerStrip)->setAxisTitle("CMSubNoise (ADC) vs Strip Number",1);

      hid = hidmanager.createHistoId("RawNoisePerStrip","det", detid);
      local_modmes.RawNoisePerStrip = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5);
      dbe_->tag(local_modmes.RawNoisePerStrip, detid);
      (local_modmes.RawNoisePerStrip)->setAxisTitle("RawNoise(ADC) vs Strip Number",1);

      hid = hidmanager.createHistoId("CMSubNoiseProfile","det", detid);
      local_modmes.CMSubNoiseProfile = dbe_->bookProfile(hid, hid, nStrip,0.5,nStrip+0.5, 100, 0., 100.);
      dbe_->tag(local_modmes.CMSubNoiseProfile, detid);
      (local_modmes.CMSubNoiseProfile)->setAxisTitle("Mean of CMSubNoise (ADC) vs Strip Number",1);

      hid = hidmanager.createHistoId("RawNoiseProfile","det", detid);
      local_modmes.RawNoiseProfile = dbe_->bookProfile(hid, hid, nStrip,0.5,nStrip+0.5, 100, 0., 100.);
      dbe_->tag(local_modmes.RawNoiseProfile, detid);
      (local_modmes.RawNoiseProfile)->setAxisTitle("Mean of RawNoise (ADC) vs Strip Number",1);

      hid = hidmanager.createHistoId("NoisyStrips","det", detid);
      local_modmes.NoisyStrips = dbe_->book2D(hid, hid, nStrip,0.5,nStrip+0.5,6,-0.5,5.5);
      dbe_->tag(local_modmes.NoisyStrips, detid);
      (local_modmes.NoisyStrips)->setAxisTitle("Strip Number",1);
      (local_modmes.NoisyStrips)->setAxisTitle("Flag Value",2);

      hid = hidmanager.createHistoId("NoisyStripDistribution","det", detid);
      local_modmes.NoisyStripDistribution = dbe_->book1D(hid, hid, 11, -0.5,10.5);
      dbe_->tag(local_modmes.NoisyStripDistribution, detid);
      (local_modmes.NoisyStripDistribution)->setAxisTitle("Flag Value",1);

      //Common Mode histos
      hid = hidmanager.createHistoId("CMDistribution","det", detid);
      local_modmes.CMDistribution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 150, -15., 15.); 
      dbe_->tag(local_modmes.CMDistribution, detid);
      (local_modmes.CMDistribution)->setAxisTitle("Common Mode (ADC) vs APV Number",1);

      hid = hidmanager.createHistoId("CMSlopeDistribution","det", detid);
      local_modmes.CMSlopeDistribution = dbe_->book2D(hid, hid, napvs,-0.5,napvs-0.5, 100, -0.05, 0.05); 
      dbe_->tag(local_modmes.CMSlopeDistribution, detid);
      (local_modmes.CMSlopeDistribution)->setAxisTitle("Common Mode Slope vs APV Number",1);

      // data from CondDB

      //Pedestals histos
      hid = hidmanager.createHistoId("PedestalFromCondDB","det", detid);
      local_modmes.PedsPerStripDB = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5); //to modify the size binning 
      dbe_->tag(local_modmes.PedsPerStripDB, detid);
      (local_modmes.PedsPerStripDB)->setAxisTitle("Pedestal from CondDB(ADC) vs Strip Number",1);

      hid = hidmanager.createHistoId("NoiseFromCondDB","det", detid);
      local_modmes.CMSubNoisePerStripDB = dbe_->book1D(hid, hid, nStrip,0.5,nStrip+0.5);
      dbe_->tag(local_modmes.CMSubNoisePerStripDB, detid);
      (local_modmes.CMSubNoisePerStripDB)->setAxisTitle("CMSubNoise from CondDB(ADC) vs Strip Number",1);

      hid = hidmanager.createHistoId("BadStripFlagCondDB","det", detid);
      local_modmes.NoisyStripsDB = dbe_->book2D(hid, hid, nStrip,0.5,nStrip+0.5,6,-0.5,5.5);
      dbe_->tag(local_modmes.NoisyStripsDB, detid);
      (local_modmes.NoisyStripsDB)->setAxisTitle("Strip Flag from CondDB(ADC) vs Strip Number",1);
    
      // append to PedMEs
      PedMEs.insert( std::make_pair(detid, local_modmes));
    } //newDetId
          
  }
  std::cout <<"Number of DETS "<<SelectedDetIds.size()<<std::endl;
}


// ------------ method called to produce the data  ------------
void SiStripMonitorPedestals::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  std::cout << "Run " << iEvent.id().run() << " Event " << iEvent.id().event() << std::endl;

   //get gain correction ES handle
  edm::ESHandle<SiStripPedestals> pedestalHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  
  if (firstEvent){
    iSetup.get<SiStripPedestalsRcd>().get(pedestalHandle);
    iSetup.get<SiStripNoisesRcd>().get(noiseHandle);
  }
  iSetup.get<SiStripDetCablingRcd>().get( detcabling );

  //Increment # of Events
  nEvTot_++;
 
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
  

  for (std::map<uint32_t, ModMEs >::const_iterator i = PedMEs.begin() ; i!=PedMEs.end() ; i++) {
    uint32_t detid = i->first; ModMEs local_modmes = i->second;
    if (firstEvent){
      //edm::LogInfo("SiStripMonitorPedestals") 
      std::cout << "[SiStripMonitorPedestals::analyze] Get Ped/Noise/Bad Strips from CondDb for DetId " << detid << std::endl;
      int nStrip  = detcabling->nApvPairs(detid) * 256;
      // Get range of pedestal and noise for the detid
      SiStripNoises::Range noiseRange = noiseHandle->getRange(detid);
      SiStripPedestals::Range pedRange = pedestalHandle->getRange(detid);

      for(int istrip=0;istrip<nStrip;++istrip){
	try{
	  //Fill Pedestals
          (local_modmes.PedsPerStripDB)->Fill(istrip+1,pedestalHandle->getPed(istrip,pedRange));
	}
	catch(cms::Exception& e){
	  edm::LogError("SiStripMonitorPedestals") << "[SiStripMonitorPedestals::analyze]  cms::Exception accessing SiStripPedestalsService_.getPedestal("<<detid<<","<<istrip<<") :  "  << " " << e.what() ;
	}
 	try{
	   //Fill Noises
         (local_modmes.CMSubNoisePerStripDB)->Fill(istrip+1,noiseHandle->getNoise(istrip,noiseRange));

	}
	catch(cms::Exception& e){
	  edm::LogError("SiStripMonitorPedestals") << "[SiStripMonitorPedestals::analyze]  cms::Exception accessing SiStripNoiseService_.getNoise("<<detid<<","<<istrip<<") :  "  << " " << e.what() ;
	}
 	try{
	    //Fill BadStripsNoise
          (local_modmes.NoisyStripsDB)->Fill(istrip+1,noiseHandle->getDisable(istrip,noiseRange)?1.:0.);


	}
	catch(cms::Exception& e){
	  edm::LogError("SiStripMonitorPedestals") << "[SiStripMonitorPedestals::analyze]  cms::Exception accessing SiStripNoiseService_.getDisable("<<detid<<","<<istrip<<") :  "  << " " << e.what() ;
	}
      }//close istrip loop
    } // firstEvent

    
    // get iterators for digis belonging to one DetId, it is an iterator, i.e. one element of the vector      
    std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = digi_collection->find( detid );
    if (digis == digi_collection->end() ||
        digis->data.size() == 0 || 
        digis->data.size() > 768) {
      if (digis == digi_collection->end()) {
        std::cout <<  " Event " <<  nEvTot_ << " DetId " <<  detid << " at the end of Digi Collection!!!"; 
      } else {
        std::cout <<  " Event " <<  nEvTot_ << " DetId " <<  detid << " # of Digis " << digis->data.size() ;
      }
      std::vector<FedChannelConnection> fed_conns = detcabling->getConnections(detid);
      for (unsigned int  k = 0; k < fed_conns.size() ; k++) {
	if (k==0) std::cout <<  " Fed Id " << fed_conns[k].fedId() << " Channel " << fed_conns[k].fedCh();
	else  std::cout <<  " Channel " << fed_conns[k].fedCh();
      }
      std::cout << std::endl;
      continue;
    }

    if ( digis->data.empty() ) { 
      edm::LogError("MonitorDigi_tmp") << "[SiStripRawDigiToRaw::createFedBuffers] Zero digis found!"; 
    } 
    uint32_t id  = detid;
    //    cout <<"Data size "<<digis->data.size()<<endl;
    apvFactory_->update(id, (*digis));
      
    if(nEvTot_ > theEventInitNumber_) {
      if(local_modmes.CMDistribution != NULL){ 
	std::vector<float> tmp;
	tmp.clear();
	apvFactory_->getCommonMode(id, tmp);
	//unpacking the info looking at the right topology
	int numberCMBlocks = int(128. / NumCMstripsInGroup_);
	int ibin=0;
	for (std::vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
	  int iapv = int (ibin/numberCMBlocks);
	  (local_modmes.CMDistribution)->Fill(iapv,static_cast<float>(*iped));
	  ibin++;
	    
	}
      }
      if(local_modmes.CMSlopeDistribution != NULL){ 
	std::vector<float> tmp;
	tmp.clear();
        int iapv = 0;
	apvFactory_->getCommonModeSlope(id, tmp);
	for (std::vector<float>::const_iterator it=tmp.begin(); it!=tmp.end();it++) {
	  (local_modmes.CMSlopeDistribution)->Fill(iapv,static_cast<float>(*it));
	  iapv++;
	}
      }
    }
      
    //asking for the status
    if((nEvTot_ - theEventInitNumber_)%theEventIterNumber_ == 1)
      {
	      
	std::vector<float> tmp;
	tmp.clear();
	apvFactory_->getPedestal(id, tmp);
	if(local_modmes.PedsPerStrip != NULL){ 
	  int numberOfApvs = int(tmp.size()/128.);
	  for(int i=0; i<numberOfApvs;i++){
	    std::vector<float> myPedPerApv;
	    apvFactory_->getPedestal(id, i, myPedPerApv);
	    float avarage = 0;
	    avarage = accumulate(myPedPerApv.begin(), myPedPerApv.end(), avarage);
	    avarage = avarage/128.;
	    (local_modmes.PedsEvolution)->setBinContent(i+1,nIteration_,avarage);
	      
	  }
	  int ibin=0;
	  
	  for (std::vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
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
	  for (std::vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
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
	  for (std::vector<float>::const_iterator iped=tmp.begin(); iped!=tmp.end();iped++) {
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
  if (firstEvent) firstEvent=false;  
}
    


void SiStripMonitorPedestals::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  if (outputMEsInRootFile) {    
    dbe_->showDirStructure();
    dbe_->save(outPutFileName);
  }
}

