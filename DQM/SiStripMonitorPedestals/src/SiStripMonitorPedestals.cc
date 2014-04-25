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
//
//

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/SiStripMonitorPedestals/interface/SiStripMonitorPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "TProfile.h"
// std
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <algorithm>

const std::string SiStripMonitorPedestals::RunMode1 = "ConDBPlotsOnly";
const std::string SiStripMonitorPedestals::RunMode2 = "CalculatedPlotsOnly";
const std::string SiStripMonitorPedestals::RunMode3 = "AllPlots";

SiStripMonitorPedestals::SiStripMonitorPedestals(edm::ParameterSet const& iConfig):
  dqmStore_(edm::Service<DQMStore>().operator->()),
  conf_(iConfig),
  pedsPSet_(iConfig.getParameter<edm::ParameterSet>("PedestalsPSet")),
  analyzed(false),
  firstEvent(true),
  signalCutPeds_(4),
  nEvTot_(0),
  nIteration_(0),
  apvFactory_(0),
  m_cacheID_(0)
{
  // retrieve producer name of input StripDigiCollection
  std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");
  std::string digiType = "VirginRaw";
  digiToken_ = consumes<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(digiProducer,digiType) ); 

  edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals  " 
					  << " Constructing....... ";     
}
//
// -- Destructor
//
SiStripMonitorPedestals::~SiStripMonitorPedestals()
{
  edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals  " 
					  << " Destructing...... ";     
  if (apvFactory_) {delete apvFactory_;} 
}
//
// -- Begin Job
//
void SiStripMonitorPedestals::beginJob() {
  theEventInitNumber_ = pedsPSet_.getParameter<int>("NumberOfEventsForInit");
  theEventIterNumber_ = pedsPSet_.getParameter<int>("NumberOfEventsForIteration");
  NumCMstripsInGroup_ = pedsPSet_.getParameter<int>("NumCMstripsInGroup");
        runTypeFlag_  = conf_.getParameter<std::string>("RunTypeFlag");
}
//
// -- BeginRun
//

void SiStripMonitorPedestals::bookHistograms(DQMStore::IBooker & ibooker, const edm::Run & run, const edm::EventSetup & eSetup)
{

  unsigned long long cacheID = eSetup.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;
    eSetup.get<SiStripDetCablingRcd>().get( detcabling );
    edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::bookHistograms: "
					    << " Creating MEs for new Cabling ";
    createMEs( ibooker , eSetup);
  } else {
    edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::bookHistograms: "
					    << " Resetting MEs ";
    for (std::map<uint32_t, ModMEs >::const_iterator idet = PedMEs.begin() ; idet!=PedMEs.end() ; idet++) {
      resetMEs(idet->first);
    }
  }
  if (runTypeFlag_ == RunMode1 || runTypeFlag_ == RunMode3 ) fillCondDBMEs(eSetup);
}


//
// -- Create Monitor Elements
//
void SiStripMonitorPedestals::createMEs(DQMStore::IBooker & ibooker , const edm::EventSetup& es) {

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  std::vector<uint32_t> SelectedDetIds;
  
  //ApvAnalysisFactory
  if (apvFactory_) {delete apvFactory_;} 
  apvFactory_ = new ApvAnalysisFactory(pedsPSet_);
  
  detcabling->addActiveDetectorsRawIds(SelectedDetIds);
  
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  // create SiStripFolderOrganizer
  SiStripFolderOrganizer folder_organizer;
  
  edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::createMEs: " 
	  << "Number of Detector Present in cabling " << SelectedDetIds.size();

  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){
    
    uint32_t detid = *idetid;

    // Check consistency in DetId
    if (detid == 0 || detid == 0xFFFFFFFF){
      edm::LogError("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::createMEs: " 
        << "Wrong DetId !!!!!! " <<  detid << " Neglecting !!!!!! ";
      continue;
    }

    unsigned int apv_pairs = detcabling->nApvPairs(detid);
    
    // Check consistency in Apv numbers
    if (apv_pairs < 1 || apv_pairs > 3) {
      edm::LogError("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::createMEs: Wrong APV Pairs  => detId "
                << detid  << " APV pairs " << apv_pairs <<  " Neglecting !!!!!! ";
      continue;
    }
    unsigned int napvs = apv_pairs * 2;
    unsigned int nStrip  = napvs * 128;

    bool newDetId =   apvFactory_->instantiateApvs(detid,napvs);  
  
    if( newDetId ) {
      ModMEs local_modmes;
      local_modmes.PedsPerStrip = 0;
      local_modmes.PedsDistribution = 0;      
      local_modmes.PedsEvolution = 0;
      local_modmes.CMSubNoisePerStrip = 0;
      local_modmes.RawNoisePerStrip = 0;
      local_modmes.CMSubNoiseProfile = 0;
      local_modmes.RawNoiseProfile = 0;
      local_modmes.NoisyStrips = 0;
      local_modmes.NoisyStripDistribution = 0;
      local_modmes.CMDistribution = 0;
      local_modmes.CMSlopeDistribution = 0;
      local_modmes.PedsPerStripDB = 0;
      local_modmes.CMSubNoisePerStripDB = 0;
      local_modmes.BadStripsDB = 0;

      std::string hid;
      // set appropriate folder using SiStripFolderOrganizer
      folder_organizer.setDetectorFolder(detid, tTopo); // pass the detid to this method
      
      // if the deid already exists in the map, then reset MEs otherwise create them
      resetMEs(detid);
      
      if (runTypeFlag_ == RunMode1 || runTypeFlag_ == RunMode3 ) {
	//Pedestals histos
	hid = hidmanager.createHistoId("PedestalFromCondDB","det", detid);
	local_modmes.PedsPerStripDB = ibooker.book1D(hid, hid, nStrip,0.5,nStrip+0.5); //to modify the size binning 
	ibooker.tag(local_modmes.PedsPerStripDB, detid);
	(local_modmes.PedsPerStripDB)->setAxisTitle("Pedestal from CondDB(ADC) vs Strip Number",1);
	
	hid = hidmanager.createHistoId("NoiseFromCondDB","det", detid);
	local_modmes.CMSubNoisePerStripDB = ibooker.book1D(hid, hid, nStrip,0.5,nStrip+0.5);
	ibooker.tag(local_modmes.CMSubNoisePerStripDB, detid);
	(local_modmes.CMSubNoisePerStripDB)->setAxisTitle("CMSubNoise from CondDB(ADC) vs Strip Number",1);
	
	hid = hidmanager.createHistoId("BadStripFlagCondDB","det", detid);
	local_modmes.BadStripsDB = ibooker.book2D(hid, hid, nStrip,0.5,nStrip+0.5,6,-0.5,5.5);
	ibooker.tag(local_modmes.BadStripsDB, detid);
	(local_modmes.BadStripsDB)->setAxisTitle("Strip Flag from CondDB(ADC) vs Strip Number",1);
      }
      if (runTypeFlag_ == RunMode2 || runTypeFlag_ == RunMode3 ) { 
	//Pedestals histos
	hid = hidmanager.createHistoId("PedsPerStrip","det", detid);
	local_modmes.PedsPerStrip = ibooker.book1D(hid, hid, nStrip,0.5,nStrip+0.5); //to modify the size binning 
	ibooker.tag(local_modmes.PedsPerStrip, detid);
	(local_modmes.PedsPerStrip)->setAxisTitle("Pedestal (ADC)  vs Strip Number ",1);
	
	hid = hidmanager.createHistoId("PedsDistribution","det", detid);
	local_modmes.PedsDistribution = ibooker.book2D(hid, hid, napvs,-0.5,napvs-0.5, 300, 200, 500); //to modify the size binning 
	ibooker.tag(local_modmes.PedsDistribution, detid);
	(local_modmes.PedsDistribution)->setAxisTitle("Apv Number",1);
	(local_modmes.PedsDistribution)->setAxisTitle("Mean Pedestal Value (ADC)",2);
	
	hid = hidmanager.createHistoId("PedsEvolution","det", detid);
	local_modmes.PedsEvolution = ibooker.book2D(hid, hid, napvs,-0.5,napvs-0.5, 50, 0., 50.); //to modify the size binning 
	ibooker.tag(local_modmes.PedsEvolution, detid);
	(local_modmes.PedsEvolution)->setAxisTitle("Apv Number",1);
	(local_modmes.PedsEvolution)->setAxisTitle("Iteration Number",2);
	
	//Noise histos
	hid = hidmanager.createHistoId("CMSubNoisePerStrip","det", detid);
	local_modmes.CMSubNoisePerStrip = ibooker.book1D(hid, hid, nStrip,0.5,nStrip+0.5);
	ibooker.tag(local_modmes.CMSubNoisePerStrip, detid);
	(local_modmes.CMSubNoisePerStrip)->setAxisTitle("CMSubNoise (ADC) vs Strip Number",1);
	
	hid = hidmanager.createHistoId("RawNoisePerStrip","det", detid);
	local_modmes.RawNoisePerStrip = ibooker.book1D(hid, hid, nStrip,0.5,nStrip+0.5);
	ibooker.tag(local_modmes.RawNoisePerStrip, detid);
	(local_modmes.RawNoisePerStrip)->setAxisTitle("RawNoise(ADC) vs Strip Number",1);
	
	hid = hidmanager.createHistoId("CMSubNoiseProfile","det", detid);
	local_modmes.CMSubNoiseProfile = ibooker.bookProfile(hid, hid, nStrip,0.5,nStrip+0.5, 100, 0., 100.);
	ibooker.tag(local_modmes.CMSubNoiseProfile, detid);
	(local_modmes.CMSubNoiseProfile)->setAxisTitle("Mean of CMSubNoise (ADC) vs Strip Number",1);
	
	hid = hidmanager.createHistoId("RawNoiseProfile","det", detid);
	local_modmes.RawNoiseProfile = ibooker.bookProfile(hid, hid, nStrip,0.5,nStrip+0.5, 100, 0., 100.);
	ibooker.tag(local_modmes.RawNoiseProfile, detid);
	(local_modmes.RawNoiseProfile)->setAxisTitle("Mean of RawNoise (ADC) vs Strip Number",1);
	
	hid = hidmanager.createHistoId("NoisyStrips","det", detid);
	local_modmes.NoisyStrips = ibooker.book2D(hid, hid, nStrip,0.5,nStrip+0.5,6,-0.5,5.5);
	ibooker.tag(local_modmes.NoisyStrips, detid);
	(local_modmes.NoisyStrips)->setAxisTitle("Strip Number",1);
	(local_modmes.NoisyStrips)->setAxisTitle("Flag Value",2);
	
	hid = hidmanager.createHistoId("NoisyStripDistribution","det", detid);
	local_modmes.NoisyStripDistribution = ibooker.book1D(hid, hid, 11, -0.5,10.5);
	ibooker.tag(local_modmes.NoisyStripDistribution, detid);
	(local_modmes.NoisyStripDistribution)->setAxisTitle("Flag Value",1);
	
	//Common Mode histos
	hid = hidmanager.createHistoId("CMDistribution","det", detid);
	local_modmes.CMDistribution = ibooker.book2D(hid, hid, napvs,-0.5,napvs-0.5, 150, -15., 15.); 
	ibooker.tag(local_modmes.CMDistribution, detid);
	(local_modmes.CMDistribution)->setAxisTitle("Common Mode (ADC) vs APV Number",1);
      
	hid = hidmanager.createHistoId("CMSlopeDistribution","det", detid);
	local_modmes.CMSlopeDistribution = ibooker.book2D(hid, hid, napvs,-0.5,napvs-0.5, 100, -0.05, 0.05); 
	ibooker.tag(local_modmes.CMSlopeDistribution, detid);
	(local_modmes.CMSlopeDistribution)->setAxisTitle("Common Mode Slope vs APV Number",1);
	
      }
      // data from CondDB
      // append to PedMEs
      PedMEs.insert( std::make_pair(detid, local_modmes));
    } //newDetId
          
  }
  edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::createMEs: Number of DETS used "
                   <<PedMEs.size();
}
// ------------ method called to produce the data  ------------
void SiStripMonitorPedestals::analyze(const edm::Event& iEvent, const edm::EventSetup& eSetup)
{

  edm::LogInfo("SiStripMonitorPedestals") <<
               "SiStripMonitorPedestals::analyze: Run " << 
               iEvent.id().run()  << " Event " << iEvent.id().event();

  eSetup.get<SiStripDetCablingRcd>().get( detcabling );

  if (runTypeFlag_ != RunMode2 &&  runTypeFlag_ != RunMode3 ) return;

  //Increment # of Events
  nEvTot_++;
 
  // get DigiCollection object from Event
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > digi_collection;
  //you have a collection as there are all the digis for the event for every detector
  iEvent.getByToken(digiToken_, digi_collection);

  //Increase the number of iterations ...
  if((nEvTot_ - theEventInitNumber_)%theEventIterNumber_ == 1) nIteration_++;
  

  // loop over all MEs
  for (std::map<uint32_t, ModMEs >::const_iterator i = PedMEs.begin() ; i!=PedMEs.end() ; i++) {
    uint32_t detid = i->first; ModMEs local_modmes = i->second;
    // get iterators for digis belonging to one DetId, it is an iterator, i.e. one element of the vector      
    std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = digi_collection->find( detid );
    if (digis == digi_collection->end() ||
        digis->data.size() == 0 || 
        digis->data.size() > 768) {
      if (digis == digi_collection->end()) {
        edm::LogError("SiStripMonitorPedestals") << " SiStripMonitorPedestals::analyze: Event " <<  nEvTot_ 
               << " DetId " <<  detid << " at the end of Digi Collection!!!"; 
      } else {
        edm::LogError("SiStripMonitorPedestals") << " [SiStripMonitorPedestals::analyze: Event " <<  nEvTot_ 
               << " DetId " <<  detid << " # of Digis " << digis->data.size() ;
      }
      std::vector<const FedChannelConnection *> fed_conns = detcabling->getConnections(detid);
      bool firstchannel(true);
      for (unsigned int  k = 0; k < fed_conns.size() ; k++) {
	if(fed_conns[k] && fed_conns[k]->isConnected()) {
	  if (firstchannel) {
	    edm::LogError("SiStripMonitorPedestals") <<" SiStripMonitorPedestals::analyze: Fed Id " <<
	      fed_conns[k]->fedId() << " Channel " << fed_conns[k]->fedCh();
	    firstchannel=false;
	  }
	  else  edm::LogError("SiStripMonitorPedestals")  <<"  SiStripMonitorPedestals::analyze: Channel " <<
		  fed_conns[k]->fedCh();
	}
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
    if((nEvTot_ - theEventInitNumber_ - theEventIterNumber_)%theEventIterNumber_ == 1)
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
//
// -- End Run
//    
void SiStripMonitorPedestals::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  if (outputMEsInRootFile) {    
    std::string outPutFileName = conf_.getParameter<std::string>("OutPutFileName");
//    dqmStore_->showDirStructure();
    dqmStore_->save(outPutFileName);
  }
}
//
// -- End Job
//    
void SiStripMonitorPedestals::endJob(void){
  edm::LogInfo("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::EndJob: " 
					  << " Finishing!! ";        
}
//
// -- Reset Monitor Elements corresponding to a detetor
//    
void SiStripMonitorPedestals::resetMEs(uint32_t idet){
  std::map<uint32_t, ModMEs >::iterator pos = PedMEs.find(idet);
  if (pos != PedMEs.end()) {
    ModMEs mod_me = pos->second;

    if (runTypeFlag_ == RunMode1 || runTypeFlag_ == RunMode3 ) { 
      mod_me.PedsPerStripDB->Reset();     
      mod_me.CMSubNoisePerStripDB->Reset();     
      mod_me.BadStripsDB->Reset();     
    }
    if (runTypeFlag_ == RunMode2 || runTypeFlag_ == RunMode3 ) { 
      mod_me.PedsPerStrip->Reset();      
      mod_me.PedsDistribution->Reset();      
      mod_me.PedsEvolution->Reset();      
      mod_me.CMSubNoisePerStrip->Reset();      
      mod_me.RawNoisePerStrip->Reset();      
      mod_me.CMSubNoiseProfile->Reset();      
      mod_me.RawNoiseProfile->Reset();      
      mod_me.NoisyStrips->Reset();      
      mod_me.CMDistribution->Reset();     
      mod_me.CMSlopeDistribution->Reset(); 
    }
  }
}
//
// -- Fill CondDB Monitoring Elements
//
void SiStripMonitorPedestals::fillCondDBMEs(edm::EventSetup const& eSetup) {
  //get Pedestal and Noise  ES handle
  edm::ESHandle<SiStripPedestals> pedestalHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  
  eSetup.get<SiStripPedestalsRcd>().get(pedestalHandle);
  eSetup.get<SiStripNoisesRcd>().get(noiseHandle);
  std::string quality_label = conf_.getParameter<std::string>("StripQualityLabel");
  eSetup.get<SiStripQualityRcd>().get(quality_label, qualityHandle);

  for (std::map<uint32_t, ModMEs >::const_iterator i = PedMEs.begin() ; i!=PedMEs.end() ; i++) {
    uint32_t detid = i->first; 
    ModMEs local_modmes = i->second;
    edm::LogInfo("SiStripMonitorPedestals") <<" SiStripMonitorPedestals::analyze: " <<
      " Get Ped/Noise/Bad Strips from CondDb for DetId " << detid;
    int nStrip  = detcabling->nApvPairs(detid) * 256;
    // Get range of pedestal and noise for the detid
    SiStripNoises::Range noiseRange = noiseHandle->getRange(detid);
    SiStripPedestals::Range pedRange = pedestalHandle->getRange(detid);
    SiStripQuality::Range qualityRange = qualityHandle->getRange(detid);
    
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
	(local_modmes.BadStripsDB)->Fill(istrip+1,qualityHandle->IsStripBad(qualityRange,istrip)?1.:0.);
	
	
      }
      catch(cms::Exception& e){
	edm::LogError("SiStripMonitorPedestals") << "[SiStripMonitorPedestals::analyze]  cms::Exception accessing SiStripNoiseService_.getDisable("<<detid<<","<<istrip<<") :  "  << " " << e.what() ;
      }
    }//close istrip loop
  }
}
DEFINE_FWK_MODULE(SiStripMonitorPedestals);
