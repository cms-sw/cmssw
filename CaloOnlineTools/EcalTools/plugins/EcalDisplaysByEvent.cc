// -*- C++ -*-
//
// Package:   EcalDisplaysByEvent 
// Class:     EcalDisplaysByEvent 
// 
/**\class EcalDisplaysByEvent EcalDisplaysByEvent.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Aug 28 5:46:22 CEST 2007
// $Id: EcalDisplaysByEvent.cc,v 1.6 2011/10/10 09:05:21 eulisse Exp $
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalDisplaysByEvent.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "TLatex.h"
#include "TLine.h"
#include "TProfile2D.h"
#include <utility> 
#include <string> 
#include <vector> 
using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//
float EcalDisplaysByEvent::gainRatio[3] = { 1., 2. , 12. }; 
edm::Service<TFileService> EcalDisplaysByEvent::fileService;

//
// constructors and destructor
//
EcalDisplaysByEvent::EcalDisplaysByEvent(const edm::ParameterSet& iConfig) :
  EBRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEB")),
  EERecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEE")),
  EBDigis_ (iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
  EEDigis_ (iConfig.getParameter<edm::InputTag>("EEDigiCollection")),
  headerProducer_ (iConfig.getParameter<edm::InputTag> ("headerProducer")),
  runNum_(-1),
  side_ (iConfig.getUntrackedParameter<int>("side", 3)),
  threshold_ (iConfig.getUntrackedParameter<double>("amplitudeThreshold", 0.5)),
  minTimingAmp_ (iConfig.getUntrackedParameter<double>("minimumTimingAmplitude", 0.100)),
  makeDigiGraphs_ (iConfig.getUntrackedParameter<bool>("makeDigiGraphs", false)),
  makeTimingHistos_ (iConfig.getUntrackedParameter<bool>("makeTimingHistos", true)),
  makeEnergyHistos_ (iConfig.getUntrackedParameter<bool>("makeEnergyHistos", true)),
  makeOccupancyHistos_ (iConfig.getUntrackedParameter<bool>("makeOccupancyHistos", true)),
  histRangeMin_ (iConfig.getUntrackedParameter<double>("histogramMinRange",0.0)),
  histRangeMax_ (iConfig.getUntrackedParameter<double>("histogramMaxRange",1.8)),
  minTimingEnergyEB_ (iConfig.getUntrackedParameter<double>("minTimingEnergyEB",0.0)),
  minTimingEnergyEE_ (iConfig.getUntrackedParameter<double>("minTimingEnergyEE",0.0))
{
  vector<int> listDefaults;
  listDefaults.push_back(-1);
  
  maskedChannels_ = iConfig.getUntrackedParameter<vector<int> >("maskedChannels", listDefaults);
  maskedFEDs_ = iConfig.getUntrackedParameter<vector<int> >("maskedFEDs", listDefaults);
  seedCrys_ = iConfig.getUntrackedParameter<vector<int> >("seedCrys",listDefaults);

  vector<string> defaultMaskedEBs;
  defaultMaskedEBs.push_back("none");
  maskedEBs_ =  iConfig.getUntrackedParameter<vector<string> >("maskedEBs",defaultMaskedEBs);
  
  fedMap_ = new EcalFedMap();

  string title1 = "Jitter for all FEDs all events";
  string name1 = "JitterAllFEDsAllEvents";
  allFedsTimingHist_ = fileService->make<TH1F>(name1.c_str(),title1.c_str(),150,-7,7);

  // load up the maskedFED list with the proper FEDids
  if(maskedFEDs_[0]==-1)
  {
    //if "actual" EB id given, then convert to FEDid and put in listFEDs_
    if(maskedEBs_[0] != "none")
    {
      maskedFEDs_.clear();
      for(vector<string>::const_iterator ebItr = maskedEBs_.begin(); ebItr != maskedEBs_.end(); ++ebItr)
      {
        maskedFEDs_.push_back(fedMap_->getFedFromSlice(*ebItr));
      }
    }
  }
  
  for (int i=0; i<10; i++)        
    abscissa[i] = i;
  
  naiveEvtNum_ = 0;

  initAllEventHistos();
}


EcalDisplaysByEvent::~EcalDisplaysByEvent()
{
}


//
// member functions
//

// ------------ method called once each job just before starting event loop  ------------
void 
EcalDisplaysByEvent::beginRun(edm::Run const &, edm::EventSetup const & c)
{
  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  ecalElectronicsMap_ = handle.product();
}

// ------------ method called to for each event  ------------
void
EcalDisplaysByEvent::analyze(edm::Event const & iEvent, edm::EventSetup const & iSetup)
{

  // get the headers
  // (one header for each supermodule)
  edm::Handle<EcalRawDataCollection> DCCHeaders;
  iEvent.getByLabel(headerProducer_, DCCHeaders);

  for (EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();
		  headerItr != DCCHeaders->end (); 
		  ++headerItr) 
  {
    FEDsAndDCCHeaders_[headerItr->id()+600] = *headerItr;
  }

  int ievt = iEvent.id().event();
  naiveEvtNum_++;

  if(runNum_==-1)
  {
    runNum_ = iEvent.id().run();
    canvasNames_ = fileService->make<TTree>("canvasNames","Names of written canvases");
    names = new std::vector<string>();
    canvasNames_->Branch("names","vector<string>",&names);
    
    histoCanvasNames_ = fileService->make<TTree>("histoCanvasNames","Names of written canvases with histos");
    histoCanvasNamesVector = new std::vector<string>();
    histoCanvasNames_->Branch("histoCanvasNamesVector","vector<string>",&histoCanvasNamesVector);
  }

  //We only want the 3x3's for this event...
  listEBChannels.clear();
  listEEChannels.clear();

  //Get hits, digis, caloTopology from event/eventSetup
  Handle<EcalRecHitCollection> EBhits;
  Handle<EcalRecHitCollection> EEhits;
  ESHandle<CaloTopology> caloTopo;
  iSetup.get<CaloTopologyRecord>().get(caloTopo);
  iEvent.getByLabel(EBRecHitCollection_, EBhits);
  iEvent.getByLabel(EERecHitCollection_, EEhits);
  iEvent.getByLabel(EBDigis_, EBdigisHandle);
  iEvent.getByLabel(EEDigis_, EEdigisHandle);

  // Initialize histos for this event
  initEvtByEvtHists(naiveEvtNum_, ievt);
  
  bool hasEBdigis = false;
  bool hasEEdigis = false;
  if(EBdigisHandle->size() > 0)
    hasEBdigis = true;
  if(EEdigisHandle->size() > 0)
    hasEEdigis = true;

  // Produce the digi graphs
  if(makeDigiGraphs_)
  {
    if(hasEBdigis) //if event has digis, it should have hits
      selectHits(EBhits, ievt, caloTopo);
    if(hasEEdigis)
      selectHits(EEhits, ievt, caloTopo);
  }

  // Produce the histos
  if(hasEBdigis)
  {
    makeHistos(EBdigisHandle);
    makeHistos(EBhits);
  }
  if(hasEEdigis)
  {
    makeHistos(EEdigisHandle);
    makeHistos(EEhits);
  }

  if(hasEBdigis || hasEEdigis)
    drawHistos();

  deleteEvtByEvtHists();
}


void EcalDisplaysByEvent::selectHits(Handle<EcalRecHitCollection> hits,
    int ievt, ESHandle<CaloTopology> caloTopo)
{
  for (EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr)
  {
    EcalRecHit hit = (*hitItr);
    DetId det = hit.id();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(det);
    int FEDid = 600+elecId.dccId();
    bool isBarrel = true;
    if(FEDid < 610 || FEDid > 645)
      isBarrel = false;
    int cryIndex = isBarrel ? ((EBDetId)det).hashedIndex() : getEEIndex(elecId);
    int ic = isBarrel ? ((EBDetId)det).ic() : getEEIndex(elecId);
    
    float ampli = hit.energy();

    vector<int>::iterator result;
    result = find(maskedFEDs_.begin(), maskedFEDs_.end(), FEDid);
    if(result != maskedFEDs_.end())
    {
      //LogWarning("EcalDisplaysByEvent") << "skipping uncalRecHit for FED " << FEDid << " ; amplitude " << ampli;
      continue;
    }      
    result = find(maskedChannels_.begin(), maskedChannels_.end(), cryIndex);
    if  (result != maskedChannels_.end())
    {
      //LogWarning("EcalDisplaysByEvent") << "skipping uncalRecHit for channel: " << cryIndex << " in fed: " << FEDid << " with amplitude " << ampli ;
      continue;
    } 
    bool cryIsInList = false;
    result = find(seedCrys_.begin(), seedCrys_.end(), cryIndex);
    if  (result != seedCrys_.end())
      cryIsInList = true;

    // Either we must have a user-requested cry (in which case there is no amplitude selection)
    // Or we pick all crys that pass the amplitude cut (in which case there is no fixed crystal selection)
    if(cryIsInList || (seedCrys_.empty() && ampli > threshold_))
    {
      // We have a winner!
      crysAndAmplitudesMap_[cryIndex] = ampli;
      string name = "Digis_Event" + intToString(naiveEvtNum_) + "_ic" + intToString(ic)
        + "_FED" + intToString(FEDid);
      string title = "Digis";
      string seed = "ic" + intToString(ic) + "_FED" + intToString(FEDid);
      int freq=1;
      pair<map<string,int>::iterator,bool> pair = seedFrequencyMap_.insert(make_pair(seed,freq));
      if(!pair.second)
      {
        ++(pair.first->second);
      }
      
      //TODO: move this also to TFileService
      TCanvas can(name.c_str(),title.c_str(),200,50,900,900);
      can.Divide(side_,side_);
      TGraph* myGraph;

      CaloNavigator<DetId> cursor = CaloNavigator<DetId>(det,caloTopo->getSubdetectorTopology(det));
      //Now put each graph in one by one
      for(int j=side_/2; j>=-side_/2; --j)
      {
        for(int i=-side_/2; i<=side_/2; ++i)
        {
          cursor.home();
          cursor.offsetBy(i,j);
          can.cd(i+1+side_/2+side_*(1-j));
          myGraph = selectDigi(*cursor,ievt);
          myGraph->Draw("A*");
        }
      }
      can.Write();
      names->push_back(name);
    }
  }
}

TGraph* EcalDisplaysByEvent::selectDigi(DetId thisDet, int ievt)
{
  int emptyY[10];
  for (int i=0; i<10; i++)
    emptyY[i] = 0;
  TGraph* emptyGraph = fileService->make<TGraph>(10, abscissa, emptyY);
  emptyGraph->SetTitle("NOT ECAL");
  
  //If the DetId is not from Ecal, return
  if(thisDet.det() != DetId::Ecal)
    return emptyGraph;
  
  emptyGraph->SetTitle("NO DIGIS");
  //find digi we need  -- can't get find() to work; need DataFrame(DetId det) to work? 
  EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(thisDet);
  int FEDid = 600+elecId.dccId();
  bool isBarrel = true;
  if(FEDid < 610 || FEDid > 645)
    isBarrel = false;
  int cryIndex = isBarrel ? ((EBDetId)thisDet).hashedIndex() : getEEIndex(elecId);
  int ic = isBarrel ? ((EBDetId)thisDet).ic() : cryIndex;

  string sliceName = fedMap_->getSliceFromFed(FEDid);
  EcalDataFrame df;
  if(isBarrel)
  {
    EBDigiCollection::const_iterator digiItr = EBdigisHandle->begin();
    while(digiItr != EBdigisHandle->end() && ((*digiItr).id() != (EBDetId)thisDet))
    {
      ++digiItr;
    }
    if(digiItr==EBdigisHandle->end())
    {
      //LogWarning("EcalDisplaysByEvent") << "Cannot find digi for ic:" << ic
      //  << " FED:" << FEDid << " evt:" << naiveEvtNum_;
      return emptyGraph;
    }
    else
      df = *digiItr;
  }
  else
  {
    EEDigiCollection::const_iterator digiItr = EEdigisHandle->begin();
    while(digiItr != EEdigisHandle->end() && ((*digiItr).id() != (EEDetId)thisDet))
    {
      ++digiItr;
    }
    if(digiItr==EEdigisHandle->end())
    {
      //LogWarning("EcalDisplaysByEvent") << "Cannot find digi for ic:" << ic
      //  << " FED:" << FEDid << " evt:" << naiveEvtNum_;
      return emptyGraph;
    }
    else df = *digiItr;
  }

  int gainId = FEDsAndDCCHeaders_[FEDid].getMgpaGain();
  int gainHuman;
  if      (gainId ==1) gainHuman =12;
  else if (gainId ==2) gainHuman =6;
  else if (gainId ==3) gainHuman =1;
  else                 gainHuman =-1; 

  double pedestal = 200;

  emptyGraph->SetTitle("FIRST TWO SAMPLES NOT GAIN12");
  if(df.sample(0).gainId()!=1 || df.sample(1).gainId()!=1) return emptyGraph; //goes to the next digi
  else {
    ordinate[0] = df.sample(0).adc();
    ordinate[1] = df.sample(1).adc();
    pedestal = (double)(ordinate[0]+ordinate[1])/(double)2;
  } 


  for (int i=0; i < df.size(); ++i ) {
    if (df.sample(i).gainId() != 0)
      ordinate[i] = (int)(pedestal+(df.sample(i).adc()-pedestal)*gainRatio[df.sample(i).gainId()-1]);
    else
      ordinate[i] = 49152; //Saturation of gain1 
  }

  TGraph* oneGraph = fileService->make<TGraph>(10, abscissa, ordinate);
  string name = "Graph_ev" + intToString(naiveEvtNum_) + "_ic" + intToString(ic)
    + "_FED" + intToString(FEDid);
  string gainString = (gainId==1) ? "Free" : intToString(gainHuman);
  string title = "Event" + intToString(naiveEvtNum_) + "_lv1a" + intToString(ievt) +
    "_ic" + intToString(ic) + "_" + sliceName + "_gain" + gainString;
    
  float energy = 0;
  map<int,float>::const_iterator itr;
  itr = crysAndAmplitudesMap_.find(cryIndex);
  if(itr!=crysAndAmplitudesMap_.end())
  {
    //edm::LogWarning("EcalDisplaysByEvent")<< "itr->second(ampli)="<< itr->second;
    energy = (float) itr->second;
  }
  //else
  //edm::LogWarning("EcalDisplaysByEvent") << "cry " << ic << "not found in ampMap";

  title+="_Energy"+floatToString(round(energy*1000));

  oneGraph->SetTitle(title.c_str());
  oneGraph->SetName(name.c_str());
  oneGraph->GetXaxis()->SetTitle("sample");
  oneGraph->GetYaxis()->SetTitle("ADC");
  return oneGraph;
}

int EcalDisplaysByEvent::getEEIndex(EcalElectronicsId elecId)
{
  int FEDid = 600+elecId.dccId();
  return 10000*FEDid+100*elecId.towerId()+5*(elecId.stripId()-1)+elecId.xtalId();
}

void EcalDisplaysByEvent::makeHistos(Handle<EBDigiCollection> ebDigiHandle) {
   const EBDigiCollection* ebDigis = ebDigiHandle.product();
   for(EBDigiCollection::const_iterator digiItr = ebDigis->begin(); digiItr != ebDigis->end(); ++digiItr) {
      EBDetId digiId = digiItr->id();
      int ieta = digiId.ieta();
      int iphi = digiId.iphi();
      digiOccupancyEBAll_->Fill(iphi,ieta);
      digiOccupancyEBcoarseAll_->Fill(iphi,ieta);
      if(makeOccupancyHistos_)
      {
        digiOccupancyEB_->Fill(iphi,ieta);
        digiOccupancyEBcoarse_->Fill(iphi,ieta);
      }
   }
}

void EcalDisplaysByEvent::makeHistos(Handle<EEDigiCollection> eeDigiHandle) {
   const EEDigiCollection* eeDigis = eeDigiHandle.product();
   for(EEDigiCollection::const_iterator digiItr = eeDigis->begin(); digiItr != eeDigis->end(); ++digiItr) {
      DetId det = digiItr->id();
      EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(det);
      size_t FEDid = 600+elecId.dccId();
      bool isEEM = false;
      if(FEDid < 610)
	isEEM = true;
      EEDetId digiId = digiItr->id();
      int ieta = digiId.iy();
      int iphi = digiId.ix();
      if(isEEM)
      {
	 digiOccupancyEEMAll_->Fill(iphi,ieta);
	 digiOccupancyEEMcoarseAll_->Fill(iphi,ieta);
         if(makeOccupancyHistos_)
         {
           digiOccupancyEEMcoarse_->Fill(iphi,ieta);
           digiOccupancyEEM_->Fill(iphi,ieta);
         }
      }
      else
      {
	 digiOccupancyEEPAll_->Fill(iphi,ieta);
	 digiOccupancyEEPcoarseAll_->Fill(iphi,ieta);
         if(makeOccupancyHistos_)
         {
           digiOccupancyEEP_->Fill(iphi,ieta);
           digiOccupancyEEPcoarse_->Fill(iphi,ieta);
         }
      }  
   }
}

void EcalDisplaysByEvent::makeHistos(Handle<EcalRecHitCollection> hits)
{
  for (EcalRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr)
  {
    EcalRecHit hit = (*hitItr);
    DetId det = hit.id();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(det);
    int FEDid = 600+elecId.dccId();
    bool isBarrel = true;
    bool isEEM = false;
    if(FEDid < 610 || FEDid > 645)
    {
      isBarrel = false;
      if(FEDid < 610)
        isEEM = true;
    }
    int iphi = isBarrel ? ((EBDetId)det).iphi() : ((EEDetId)det).ix();
    int ieta = isBarrel ? ((EBDetId)det).ieta() : ((EEDetId)det).iy();
    float energy = hit.energy();
    float time = hit.time();
    
    // Fill energy histos
    if(makeEnergyHistos_)
    {
      if(isBarrel)
      {
        energyEB_->Fill(energy);
        energyMapEB_->Fill(iphi,ieta,energy);
        energyMapEBcoarse_->Fill(iphi,ieta,energy);
      }
      else if(isEEM)
      {
        energyEEM_->Fill(energy);
        energyMapEEM_->Fill(iphi,ieta,energy);
        energyMapEEMcoarse_->Fill(iphi,ieta,energy);
      }
      else
      {
        energyEEP_->Fill(energy);
        energyMapEEP_->Fill(iphi,ieta,energy);
        energyMapEEPcoarse_->Fill(iphi,ieta,energy);
      }
    }
    // Fill occupancy histos
    if(makeOccupancyHistos_)
    {
      if(isBarrel)
      {
        recHitOccupancyEB_->Fill(iphi,ieta);
        recHitOccupancyEBcoarse_->Fill(iphi,ieta);
      }
      else if(isEEM)
      {
	 recHitOccupancyEEM_->Fill(iphi,ieta);
	 recHitOccupancyEEMcoarse_->Fill(iphi,ieta);
      }
      else
      {
	 recHitOccupancyEEP_->Fill(iphi,ieta);
	 recHitOccupancyEEPcoarse_->Fill(iphi,ieta);
      }
    }

    // Fill timing histos
    if(makeTimingHistos_)
    {
      if(isBarrel) {
        timingEB_->Fill(time);
	if(energy > minTimingEnergyEB_) {
	   timingMapEB_->Fill(iphi,ieta,time);
	   //timingMapEBCoarse_->Fill(iphi,ieta,time);
	}
      }
      else if(isEEM) {
	 timingEEM_->Fill(time);
	 if(energy > minTimingEnergyEE_)
         {
	    timingMapEEM_->Fill(iphi,ieta,time);
	    //timingMapEEMCoarse_->Fill(iphi,ieta,time);
         }
      }
      else {
	 timingEEP_->Fill(time);
	 if(energy > minTimingEnergyEE_)
         {
	    timingMapEEP_->Fill(iphi,ieta,time);
	    //timingMapEEPCoarse_->Fill(iphi,ieta,time);
         }
      }
    }

    //All events
    if(isBarrel)
    {
      energyEBAll_->Fill(energy);
      energyMapEBAll_->Fill(iphi,ieta,energy);
      energyMapEBcoarseAll_->Fill(iphi,ieta,energy);
      recHitOccupancyEBAll_->Fill(iphi,ieta);
      recHitOccupancyEBcoarseAll_->Fill(iphi,ieta);
      timingEBAll_->Fill(time);
      if(energy > minTimingEnergyEB_)
      {
        timingMapEBAll_->Fill(iphi,ieta,time);
        timingMapEBCoarseAll_->Fill(iphi,ieta,time);
      }
    }
    else if(isEEM)
    {
      energyEEMAll_->Fill(energy);
      energyMapEEMAll_->Fill(iphi,ieta,energy);
      energyMapEEMcoarseAll_->Fill(iphi,ieta,energy);
      recHitOccupancyEEMAll_->Fill(iphi,ieta);
      recHitOccupancyEEMcoarseAll_->Fill(iphi,ieta);
      timingEEMAll_->Fill(time);
      if(energy > minTimingEnergyEE_)
      {
        timingMapEEMAll_->Fill(iphi,ieta,time);
        timingMapEEMCoarseAll_->Fill(iphi,ieta,time);
      }
    }
    else
    {
      energyEEPAll_->Fill(energy);
      energyMapEEPAll_->Fill(iphi,ieta,energy);
      energyMapEEPcoarseAll_->Fill(iphi,ieta,energy);
      recHitOccupancyEEPAll_->Fill(iphi,ieta);
      recHitOccupancyEEPcoarseAll_->Fill(iphi,ieta);
      timingEEPAll_->Fill(time);
      if(energy > minTimingEnergyEE_)
      {
        timingMapEEPAll_->Fill(iphi,ieta,time);
        timingMapEEPCoarseAll_->Fill(iphi,ieta,time);
      }
    }
    // Fill FED-by-FED timing histos (all events)
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(timingHist==0)
    {
      initHists(FEDid);
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    if(energy > minTimingAmp_)
    {
      timingHist->Fill(hit.time());
      allFedsTimingHist_->Fill(hit.time());
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalDisplaysByEvent::endJob()
{
  //All-event canvases
  drawCanvas(timingCanvasAll_,timingEEMAll_,timingEBAll_,timingEEPAll_);
  drawCanvas(timingMapCanvasAll_,timingMapEEMAll_,timingMapEBAll_,timingMapEEPAll_);
  drawCanvas(energyCanvasAll_,energyEEMAll_,energyEBAll_,energyEEPAll_);
  drawCanvas(energyMapCanvasAll_,energyMapEEMAll_,energyMapEBAll_,energyMapEEPAll_);
  drawCanvas(energyMapCoarseCanvasAll_,energyMapEEMcoarseAll_,energyMapEBcoarseAll_,energyMapEEPcoarseAll_);
  drawCanvas(recHitOccupancyCanvasAll_,recHitOccupancyEEMAll_,recHitOccupancyEBAll_,recHitOccupancyEEPAll_);
  drawCanvas(recHitOccupancyCoarseCanvasAll_,recHitOccupancyEEMcoarseAll_,recHitOccupancyEBcoarseAll_,recHitOccupancyEEPcoarseAll_);
  drawCanvas(digiOccupancyCanvasAll_,digiOccupancyEEMAll_,digiOccupancyEBAll_,digiOccupancyEEPAll_);
  drawCanvas(digiOccupancyCoarseCanvasAll_,digiOccupancyEEMcoarseAll_,digiOccupancyEBcoarseAll_,digiOccupancyEEPcoarseAll_);

  if(runNum_ != -1) {
    canvasNames_->Fill();
    histoCanvasNames_->Fill();
  }

  string frequencies = "";
  for(std::map<std::string,int>::const_iterator itr = seedFrequencyMap_.begin();
      itr != seedFrequencyMap_.end(); ++itr)
  {
    if(itr->second > 1)
    {
      frequencies+=itr->first;
      frequencies+=" Frequency: ";
      frequencies+=intToString(itr->second);
      frequencies+="\n";
    }
  }
  LogWarning("EcalDisplaysByEvent") << "Found seeds with frequency > 1: " << "\n\n" << frequencies;
  
  std::string channels;
  for(std::vector<int>::const_iterator itr = maskedChannels_.begin();
      itr != maskedChannels_.end(); ++itr)
  {
    channels+=intToString(*itr);
    channels+=",";
  }
  
  std::string feds;
  for(std::vector<int>::const_iterator itr = maskedFEDs_.begin();
      itr != maskedFEDs_.end(); ++itr)
  {
    feds+=intToString(*itr);
    feds+=",";
  }

  LogWarning("EcalDisplaysByEvent") << "Masked channels are: " << channels;
  LogWarning("EcalDisplaysByEvent") << "Masked FEDs are: " << feds << " and that is all!";
}

std::string EcalDisplaysByEvent::intToString(int num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}

std::string EcalDisplaysByEvent::floatToString(float num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}

// insert the hist map into the map keyed by FED number
void EcalDisplaysByEvent::initHists(int FED)
{
  using namespace std;
  
  string title1 = "Jitter for ";
  title1.append(fedMap_->getSliceFromFed(FED));
  string name1 = "JitterFED";
  name1.append(intToString(FED));
  TH1F* timingHist = fileService->make<TH1F>(name1.c_str(),title1.c_str(),150,-7,7);
  FEDsAndTimingHists_[FED] = timingHist;
}

void EcalDisplaysByEvent::initEvtByEvtHists(int naiveEvtNum_, int ievt)
{
  string lastPart = intToString(naiveEvtNum_)+"_LV1a"+intToString(ievt);
  if(makeTimingHistos_)
  {
    string canvasTitle = "Timing_Event"+lastPart;
    timingEB_ = new TH1F("timeForAllFedsEB","timeForAllFeds;Relative Time (1 clock = 25ns)",78,-7,7);
    timingEEM_ = new TH1F("timeForAllFedsEEM","timeForAllFeds_EEM;Relative Time (1 clock = 25ns)",78,-7,7);
    timingEEP_ = new TH1F("timeForAllFedsEEP","timeForAllFeds_EEP;Relative Time (1 clock = 25ns)",78,-7,7);
    timingCanvas_ = new TCanvas(canvasTitle.c_str(), canvasTitle.c_str(),300,60,500,200);
    timingMapEB_ = init3DEcalHist("TimingMap", EB_FINE);
    timingMapEEM_ = init3DEcalHist("TimingMap", EEM_FINE);
    timingMapEEP_ = init3DEcalHist("TimingMap", EEP_FINE);
    timingMapCanvas_ = init2DEcalCanvas("TimingMap_Event"+lastPart);
    //timingMapEBCoarse_ = init3DEcalHist("TimingMap", EB_COARSE);
    timingMapEEMCoarse_ = init3DEcalHist("TimingMap", EEM_COARSE);
    timingMapEEPCoarse_ = init3DEcalHist("TimingMap", EEP_COARSE);
    //timingMapCoarseCanvas_ = init2DEcalCanvas("TimingMapCoarse_Event"+lastPart);  
  }
  if(makeEnergyHistos_)
  {
    energyEB_ = new TH1F("energyEB","Energy for EB Feds (GeV)",200,histRangeMin_,histRangeMax_);
    energyEEM_ = new TH1F("energyEEM","Energy for EEM Feds (GeV)",200,histRangeMin_,10.0);
    energyEEP_ = new TH1F("energyEEP","Energy for EEP Feds (GeV)",200,histRangeMin_,10.0);
    string canvasTitle = "Energy_Event"+lastPart;
    energyCanvas_ = new TCanvas(canvasTitle.c_str(), canvasTitle.c_str(),300,60,500,200);

    // Energy map hists
    energyMapEB_            = init2DEcalHist("EnergyMap",EB_FINE);
    energyMapEBcoarse_      = init2DEcalHist("EnergyMap",EB_COARSE);
    energyMapEEMcoarse_           = init2DEcalHist("EnergyMap",EEM_COARSE);
    energyMapEEM_                 = init2DEcalHist("EnergyMap",EEM_FINE);
    energyMapEEPcoarse_           = init2DEcalHist("EnergyMap",EEP_COARSE);
    energyMapEEP_                 = init2DEcalHist("EnergyMap",EEP_FINE);
    energyMapCanvas_ = init2DEcalCanvas("EnergyMap_Event"+lastPart);
    energyMapCoarseCanvas_ = init2DEcalCanvas("EnergyMapCoarse_Event"+lastPart);
  }
  if(makeOccupancyHistos_) 
  {
     // RecHit Occupancy
    recHitOccupancyEB_ = init2DEcalHist("RecHitOccupancy",EB_FINE);
    recHitOccupancyEBcoarse_ = init2DEcalHist("RecHitOccupancy",EB_COARSE);
    recHitOccupancyEEMcoarse_ = init2DEcalHist("RecHitOccupancy",EEM_COARSE);
    recHitOccupancyEEM_ = init2DEcalHist("RecHitOccupancy",EEM_FINE);
    recHitOccupancyEEPcoarse_ = init2DEcalHist("RecHitOccupancy",EEP_COARSE);
    recHitOccupancyEEP_ = init2DEcalHist("RecHitOccupancy",EEP_FINE);
    recHitOccupancyCanvas_ = init2DEcalCanvas("RecHitOccupancy_Event"+lastPart);
    recHitOccupancyCoarseCanvas_ =init2DEcalCanvas("RecHitOccupancyCoarse_Event"+lastPart);

    //DigiOccupancy
    digiOccupancyEB_ = init2DEcalHist("DigiOccupancy",EB_FINE);
    digiOccupancyEBcoarse_ = init2DEcalHist("DigiOccupancy", EB_COARSE);
    digiOccupancyEEMcoarse_ = init2DEcalHist("DigiOccupancy",EEM_COARSE);
    digiOccupancyEEM_ = init2DEcalHist("DigiOccupancy",EEM_FINE);
    digiOccupancyEEPcoarse_ = init2DEcalHist("DigiOccupancy",EEP_COARSE);
    digiOccupancyEEP_ = init2DEcalHist("DigiOccupancy",EEP_FINE);
    digiOccupancyCanvas_ = init2DEcalCanvas("DigiOccupancy_Event"+lastPart);
    digiOccupancyCoarseCanvas_ = init2DEcalCanvas("DigiOccupancyCoarse_Event"+lastPart);
  }
}

void EcalDisplaysByEvent::deleteEvtByEvtHists()
{
    delete timingEB_;
    delete timingEEM_;
    delete timingEEP_;
    delete timingMapEB_;
    delete timingMapEEM_;
    delete timingMapEEP_;
    delete timingCanvas_;
    delete timingMapCanvas_;
    delete energyEB_;
    delete energyEEM_;
    delete energyEEP_;
    delete energyMapEB_;
    delete energyMapEEM_;
    delete energyMapEEP_;
    delete energyMapEBcoarse_;
    delete energyMapEEMcoarse_;
    delete energyMapEEPcoarse_;
    delete energyCanvas_;
    delete energyMapCanvas_;
    delete energyMapCoarseCanvas_;
    delete recHitOccupancyEB_;
    delete recHitOccupancyEEP_;
    delete recHitOccupancyEEM_;
    delete recHitOccupancyEBcoarse_;
    delete recHitOccupancyEEMcoarse_;
    delete recHitOccupancyEEPcoarse_;
    delete digiOccupancyEB_;
    delete digiOccupancyEEM_;
    delete digiOccupancyEEP_;
    delete digiOccupancyEBcoarse_;
    delete digiOccupancyEEMcoarse_;
    delete digiOccupancyEEPcoarse_;
    delete recHitOccupancyCanvas_;
    delete recHitOccupancyCoarseCanvas_;
    delete digiOccupancyCanvas_;
    delete digiOccupancyCoarseCanvas_;
}

void EcalDisplaysByEvent::initAllEventHistos()
{
  string canvasTitle = "Timing_AllEvents";
  timingEBAll_ = new TH1F("timeForAllFedsEBAll","timeForAllFeds;Relative Time (1 clock = 25ns)",78,-7,7);
  timingEEMAll_ = new TH1F("timeForAllFedsEEMAll","timeForAllFeds_EEM;Relative Time (1 clock = 25ns)",78,-7,7);
  timingEEPAll_ = new TH1F("timeForAllFedsEEPAll","timeForAllFeds_EEP;Relative Time (1 clock = 25ns)",78,-7,7);
  timingCanvasAll_ = new TCanvas(canvasTitle.c_str(), canvasTitle.c_str(),300,60,500,200);
  timingMapEBAll_ = init3DEcalHist("TimingMapA", EB_FINE);
  timingMapEEMAll_ = init3DEcalHist("TimingMapA", EEM_FINE);
  timingMapEEPAll_ = init3DEcalHist("TimingMapA", EEP_FINE);
  timingMapCanvasAll_ = init2DEcalCanvas("TimingMap_AllEvents");
  timingMapEBCoarseAll_ = init3DEcalHist("TimingMapA", EB_COARSE);
  timingMapEEMCoarseAll_ = init3DEcalHist("TimingMapA", EEM_COARSE);
  timingMapEEPCoarseAll_ = init3DEcalHist("TimingMapA", EEP_COARSE);
  //timingMapCoarseCanvasAll_ = init2DEcalCanvas("TimingMapCoarse_AllEvents"); 
  energyEBAll_ = new TH1F("energyEBAllEvents","Energy for EB Feds (GeV)",200,histRangeMin_,histRangeMax_);
  energyEEMAll_ = new TH1F("energyEEMAllEvents","Energy for EEM Feds (GeV)",200,histRangeMin_,10.0);
  energyEEPAll_ = new TH1F("energyEEPAllEvents","Energy for EEP Feds (GeV)",200,histRangeMin_,10.0);
  canvasTitle = "Energy_AllEvents";
  energyCanvasAll_ = new TCanvas(canvasTitle.c_str(), canvasTitle.c_str(),300,60,500,200);

  // Energy map hists
  energyMapEBAll_            = init2DEcalHist("EnergyMapA",EB_FINE);
  energyMapEBcoarseAll_      = init2DEcalHist("EnergyMapA",EB_COARSE);
  energyMapEEMcoarseAll_           = init2DEcalHist("EnergyMapA",EEM_COARSE);
  energyMapEEMAll_                 = init2DEcalHist("EnergyMapA",EEM_FINE);
  energyMapEEPcoarseAll_           = init2DEcalHist("EnergyMapA",EEP_COARSE);
  energyMapEEPAll_                 = init2DEcalHist("EnergyMapA",EEP_FINE);
  energyMapCanvasAll_ = init2DEcalCanvas("EnergyMap_AllEvents");
  energyMapCoarseCanvasAll_ = init2DEcalCanvas("EnergyMapCoarse_AllEvents");
  // RecHit Occupancy
  recHitOccupancyEBAll_ = init2DEcalHist("RecHitOccupancyA",EB_FINE);
  recHitOccupancyEBcoarseAll_ = init2DEcalHist("RecHitOccupancyA",EB_COARSE);
  recHitOccupancyEEMcoarseAll_ = init2DEcalHist("RecHitOccupancyA",EEM_COARSE);
  recHitOccupancyEEMAll_ = init2DEcalHist("RecHitOccupancyA",EEM_FINE);
  recHitOccupancyEEPcoarseAll_ = init2DEcalHist("RecHitOccupancyA",EEP_COARSE);
  recHitOccupancyEEPAll_ = init2DEcalHist("RecHitOccupancyA",EEP_FINE);
  recHitOccupancyCanvasAll_ = init2DEcalCanvas("RecHitOccupancy_AllEvents");
  recHitOccupancyCoarseCanvasAll_ =init2DEcalCanvas("RecHitOccupancyCoarse_AllEvents");

  //DigiOccupancy
  digiOccupancyEBAll_ = init2DEcalHist("DigiOccupancyA",EB_FINE);
  digiOccupancyEBcoarseAll_ = init2DEcalHist("DigiOccupancyA", EB_COARSE);
  digiOccupancyEEMcoarseAll_ = init2DEcalHist("DigiOccupancyA",EEM_COARSE);
  digiOccupancyEEMAll_ = init2DEcalHist("DigiOccupancyA",EEM_FINE);
  digiOccupancyEEPcoarseAll_ = init2DEcalHist("DigiOccupancyA",EEP_COARSE);
  digiOccupancyEEPAll_ = init2DEcalHist("DigiOccupancyA",EEP_FINE);
  digiOccupancyCanvasAll_ = init2DEcalCanvas("DigiOccupancy_AllEvents");
  digiOccupancyCoarseCanvasAll_ = init2DEcalCanvas("DigiOccupancyCoarse_AllEvents");

}


TH3F* EcalDisplaysByEvent::init3DEcalHist(std::string histTypeName, int subDet) {
   TH3F* hist;
   bool isBarrel = (subDet == EB_FINE || subDet == EB_COARSE) ? true : false;
   bool isCoarse = (subDet == EB_COARSE || subDet == EEM_COARSE || subDet == EEP_COARSE) ? true : false; 
   bool isEEM = (subDet == EEM_FINE || subDet == EEM_COARSE) ? true : false;
   std::string histName = histTypeName;
   std::string histTitle = histTypeName;
   double ttEtaBins[36] = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86 };
   double modEtaBins[10]={-85, -65, -45, -25, 0, 1, 26, 46, 66, 86};
   double ttPhiBins[73];
   double modPhiBins[19];
   double timingBins[79];
   for (int i = 0; i < 79; ++i)
   {
      timingBins[i]= 6 - 7. + double(i) * 14. / 78.;
      if (i<73)  
      {       
	 ttPhiBins[i]=1+5*i;
	 if ( i < 19) 
	 {       
	    modPhiBins[i]=1+20*i;
	 }       
      }       

   }

   if(isBarrel) {
      histName = histName + "EB";
      histTitle = histTitle + " EB";
      if(isCoarse) {
	 histName = histName + "Coarse";
	 histTitle = histTitle + " by Module Nominal value = 6;iphi;ieta ";
	 hist = new TH3F(histName.c_str(),histTitle.c_str(),18,modPhiBins,9,modEtaBins,78,timingBins);
      }
      else {
	 histTitle = histTitle + " by TT Nominal value = 6;iphi;ieta";
	 hist = new TH3F(histName.c_str(),histTitle.c_str(),360/5,ttPhiBins,35,ttEtaBins,78,timingBins);
      }
   }
   else {
      double ttXBins[21];
      double ttYBins[21];
      for(int i=0;i!=21;++i) {
	 ttXBins[i] = 1 + 5*i;
	 ttYBins[i] = 1 + 5*i;
      }
      if(isEEM) {
	 histName = histName + "EEM";
	 histTitle = histTitle + " EEM";
      }
      else {
	 histName = histName + "EEP";
	 histTitle = histTitle + " EEP";
      }
      if(isCoarse) {
	 histName = histName + "Coarse";
	 histTitle = histTitle + " by Module Nominal value = 6;ix;iy";
	 hist = new TH3F(histName.c_str(),histTitle.c_str(),20,ttXBins,20,ttYBins,78,timingBins);
      }
      else {
	 histTitle = histTitle + " by TT Nominal value = 6;ix;iy";
	 hist = new TH3F(histName.c_str(),histTitle.c_str(),20,ttXBins,20,ttYBins,78,timingBins);
      }
   }
   return hist;
}

TH2F* EcalDisplaysByEvent::init2DEcalHist(std::string histTypeName, int subDet) {
   TH2F* hist;
   bool isBarrel = (subDet == EB_FINE || subDet == EB_COARSE) ? true : false;
   bool isCoarse = (subDet == EB_COARSE || subDet == EEM_COARSE || subDet == EEP_COARSE) ? true : false; 
   bool isEEM = (subDet == EEM_FINE || subDet == EEM_COARSE) ? true : false;
   std::string histName = histTypeName;
   std::string histTitle = histTypeName;
   if(isBarrel) {
      histName = histName + "EB";
      histTitle = histTitle + " EB";
      if(isCoarse) {
	 histName = histName + "Coarse";
	 histTitle = histTitle + " Coarse;iphi;ieta";
	 double ttEtaBins[36] = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, 
	    -35, -30, -25, -20, -15, -10, -5, 0, 1, 6, 11, 16, 
	    21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86 };
	 hist = new TH2F(histName.c_str(),histTitle.c_str(),360/5,1,361.,35,ttEtaBins);
      }
      else {
	 histTitle = histTitle + ";iphi;ieta";
	 hist = new TH2F(histName.c_str(),histTitle.c_str(),360,1,361.,172,-86,86);
      }
   }
   else {
      if(isEEM) {
	 histName = histName + "EEM";
	 histTitle = histTitle + " EEM";
      }
      else {
	 histName = histName + "EEP";
	 histTitle = histTitle + " EEP";
      }
      if(isCoarse) {
	 histName = histName + "Coarse";
	 histTitle = histTitle + " Coarse;ix;iy";
	 hist = new TH2F(histName.c_str(),histTitle.c_str(),20,1,101,20,1,101);
      }
      else {
	 histTitle = histTitle + ";ix;iy";
	 hist = new TH2F(histName.c_str(),histTitle.c_str(),100,1,101,100,1,101);
      }
   }
   return hist;
}

TCanvas* EcalDisplaysByEvent::init2DEcalCanvas(std::string canvasTitle) {
    TCanvas* canvas = new TCanvas(canvasTitle.c_str(), canvasTitle.c_str(),300,60,500,200);
    return canvas;
}

void EcalDisplaysByEvent::drawHistos()
{
  if(makeTimingHistos_)
  {
    // Put the timing canvas together
    drawCanvas(timingCanvas_,timingEEM_,timingEB_,timingEEP_);
    //   drawCanvas(timingMapCoarseCanvas_,timingMapEEMCoarse_,timingMapEBCoarse_,timingMapEEPCoarse_);
    drawCanvas(timingMapCanvas_,timingMapEEM_,timingMapEB_,timingMapEEP_);
  }
  if(makeEnergyHistos_)
  {
    // Put the energy canvas together
    drawCanvas(energyCanvas_,energyEEM_,energyEB_,energyEEP_);
    // Put the energy map canvas together
    drawCanvas(energyMapCanvas_,energyMapEEM_,energyMapEB_,energyMapEEP_);
    drawCanvas(energyMapCoarseCanvas_,energyMapEEMcoarse_,energyMapEBcoarse_,energyMapEEPcoarse_);
  }
  if(makeOccupancyHistos_)
  {
    // Put the occupancy canvas together
    drawCanvas(recHitOccupancyCanvas_,recHitOccupancyEEM_,recHitOccupancyEB_,recHitOccupancyEEP_);
    drawCanvas(recHitOccupancyCoarseCanvas_,recHitOccupancyEEMcoarse_,recHitOccupancyEBcoarse_,recHitOccupancyEEPcoarse_);
    // And the same for the digis
    drawCanvas(digiOccupancyCanvas_,digiOccupancyEEM_,digiOccupancyEB_,digiOccupancyEEP_);
    drawCanvas(digiOccupancyCoarseCanvas_,digiOccupancyEEMcoarse_,digiOccupancyEBcoarse_,digiOccupancyEEPcoarse_);
  }
}

void EcalDisplaysByEvent::drawCanvas(TCanvas* canvas, TH1F* hist1, TH1F* hist2, TH1F* hist3) {
  canvas->Divide(1,2); 
  canvas->cd(1)->Divide(2,1); 
  canvas->cd(1)->cd(1);
  hist1->Draw();
  canvas->cd(2); 
  hist2->Draw();
  canvas->cd(1)->cd(2);
  hist3->Draw(); 
  histoCanvasNamesVector->push_back(canvas->GetName());
  canvas->SetCanvasSize(500,500);
  canvas->SetFixedAspectRatio(true);
  canvas->Write();
}

void EcalDisplaysByEvent::drawCanvas(TCanvas* canvas, TH2F* hist1, TH2F* hist2, TH2F* hist3) {
  canvas->Divide(1,2);
  canvas->cd(1)->Divide(2,1);
  // EEM
  canvas->cd(1)->cd(1);
  hist1->Draw("colz");
  drawEELines();  
  // EB
  canvas->cd(2);
  hist2->Draw("colz");
  hist2->GetXaxis()->SetNdivisions(-18);
  hist2->GetYaxis()->SetNdivisions(2);
  canvas->GetPad(2)->SetGridx(1);
  canvas->GetPad(2)->SetGridy(1);
  // EEP
  canvas->cd(1)->cd(2);
  hist3->Draw("colz");
  drawEELines();  
  histoCanvasNamesVector->push_back(canvas->GetName());
  canvas->SetCanvasSize(500,500);
  canvas->SetFixedAspectRatio(true);
  canvas->Write();
}

void EcalDisplaysByEvent::drawCanvas(TCanvas* canvas, TH3F* hist1, TH3F* hist2, TH3F* hist3) {
   if(canvas == timingMapCoarseCanvas_) {
      canvas->cd();
      TProfile2D* profile2 = (TProfile2D*) hist2->Project3DProfile("yx");
      profile2->Draw("colz");
      drawTimingErrors(profile2);
   }
   else {
      canvas->Divide(1,2);
      canvas->cd(1)->Divide(2,1);
      // EEM
      canvas->cd(1)->cd(1);
      TProfile2D* profile1 = (TProfile2D*) hist1->Project3DProfile("yx");
      profile1->Draw("colz");
      drawEELines();  
      // EB
      canvas->cd(2);
      TProfile2D* profile2 = (TProfile2D*) hist2->Project3DProfile("yx");
      profile2->Draw("colz");
      profile2->GetXaxis()->SetNdivisions(-18);
      profile2->GetYaxis()->SetNdivisions(2);
      canvas->GetPad(2)->SetGridx(1);
      canvas->GetPad(2)->SetGridy(1);
      // EEP
      canvas->cd(1)->cd(2);
      TProfile2D* profile3 = (TProfile2D*) hist3->Project3DProfile("yx");
      profile3->Draw("colz");
      drawEELines();  
   }
   histoCanvasNamesVector->push_back(canvas->GetName());
   canvas->SetCanvasSize(500,500);
   canvas->SetFixedAspectRatio(true);
   canvas->Write();
}

void EcalDisplaysByEvent::drawTimingErrors(TProfile2D* profile) {
   int nxb = profile->GetNbinsX();
   int nyb = profile->GetNbinsY();
   char tempErr[200];
   for(int i=0;i!=nxb;++i) {
      for(int j=0;j!=nyb;++j) {
	 int xb = i+1;
	 int yb = j+1;
//   std::cout << "xb: " << xb << "\tyb: " << yb << std::endl;
	 double xcorr = profile->GetBinCenter(xb);
	 double ycorr = profile->GetBinCenter(yb);
	 sprintf(tempErr,"%0.2f",profile->GetBinError(xb,yb));
	 int nBin = profile->GetBin(xb,yb,0);
	 int nBinEntries = (int) profile->GetBinEntries(nBin);
	 if(nBinEntries != 0) {
	    TLatex* tex = new TLatex(xcorr,ycorr,tempErr);
	    tex->SetTextAlign(23);
	    tex->SetTextSize(42);
	    tex->SetTextSize(0.025);
	    tex->SetLineWidth(2);
	    tex->Draw();
            delete tex;
	 }
	 sprintf(tempErr,"%i",nBinEntries);
	 if(nBinEntries!=0) {
	    TLatex* tex = new TLatex(xcorr,ycorr,tempErr);
	    tex->SetTextAlign(21);
	    tex->SetTextFont(42);
	    tex->SetTextSize(0.025);
	    tex->SetLineWidth(2);
	    tex->Draw();
            delete tex;
	 }
      }
   }
}

void EcalDisplaysByEvent::drawEELines() {

  int ixSectorsEE[202] = {61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 0,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100,  0, 61, 65, 65, 70, 70, 80, 80, 90, 90, 92,  0, 61, 65, 65, 90, 90, 97,  0, 57, 60, 60, 65, 65, 70, 70, 75, 75, 80, 80,  0, 50, 50,  0, 43, 40, 40, 35, 35, 30, 30, 25, 25, 20, 20,  0, 39, 35, 35, 10, 10,  3,  0, 39, 35, 35, 30, 30, 20, 20, 10, 10,  8,  0, 45, 45, 40, 40, 35, 35,  0, 55, 55, 60, 60, 65, 65};
 
  int iySectorsEE[202] = {50, 55, 55, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 55, 55, 45, 45, 43, 43, 42, 42, 41, 41, 40, 40, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 45, 45, 50,  0, 50, 60, 60, 65, 65, 75, 75, 80, 80, 85, 85, 87, 87, 92, 92, 95, 95, 97, 97,100,100, 97, 97, 95, 95, 92, 92, 87, 87, 85, 85, 80, 80, 75, 75, 65, 65, 60, 60, 40, 40, 35, 35, 25, 25, 20, 20, 15, 15, 13, 13,  8,  8,  5,  5,  3,  3,  0,  0,  3,  3,  5,  5,  8,  8, 13, 13, 15, 15, 20, 20, 25, 25, 35, 35, 40, 40, 50,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 50, 50, 55, 55, 60, 60,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 61,100,  0, 60, 60, 65, 65, 70, 70, 75, 75, 85, 85, 87,  0, 50, 50, 55, 55, 60, 60,  0, 45, 45, 40, 40, 35, 35, 30, 30, 25, 25,  0, 39, 30, 30, 15, 15,  5,  0, 39, 30, 30, 15, 15,  5};


 for ( int i=0; i<202; i++) {
   ixSectorsEE[i] += 1;
   iySectorsEE[i] += 1;
//   std::cout << i << " " << ixSectorsEE[i] << " " << iySectorsEE[i] << std::endl;
 }

 TLine l;
 l.SetLineWidth(1);
 for ( int i=0; i<201; i=i+1) {
   if ( (ixSectorsEE[i]!=1 || iySectorsEE[i]!=1) && 
	(ixSectorsEE[i+1]!=1 || iySectorsEE[i+1]!=1) ) {
     l.DrawLine(ixSectorsEE[i], iySectorsEE[i], 
		ixSectorsEE[i+1], iySectorsEE[i+1]);
   }
 }


}
