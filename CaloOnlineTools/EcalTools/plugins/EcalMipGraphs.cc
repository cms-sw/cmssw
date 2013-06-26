// -*- C++ -*-
//
// Package:   EcalMipGraphs 
// Class:     EcalMipGraphs 
// 
/**\class EcalMipGraphs EcalMipGraphs.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalMipGraphs.cc,v 1.12 2010/10/20 10:02:00 elmer Exp $
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalMipGraphs.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "TCanvas.h"
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
float EcalMipGraphs::gainRatio[3] = { 1., 2. , 12. }; 
edm::Service<TFileService> EcalMipGraphs::fileService;

//
// constructors and destructor
//
EcalMipGraphs::EcalMipGraphs(const edm::ParameterSet& iConfig) :
  EBRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEB")),
  EERecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEE")),
  EBDigis_ (iConfig.getParameter<edm::InputTag>("EBDigiCollection")),
  EEDigis_ (iConfig.getParameter<edm::InputTag>("EEDigiCollection")),
  headerProducer_ (iConfig.getParameter<edm::InputTag> ("headerProducer")),
  runNum_(-1),
  side_ (iConfig.getUntrackedParameter<int>("side", 3)),
  threshold_ (iConfig.getUntrackedParameter<double>("amplitudeThreshold", 12.0)),
  minTimingAmp_ (iConfig.getUntrackedParameter<double>("minimumTimingAmplitude", 0.100))
{
  vector<int> listDefaults;
  listDefaults.push_back(-1);
  
  maskedChannels_ = iConfig.getUntrackedParameter<vector<int> >("maskedChannels", listDefaults);
  maskedFEDs_ = iConfig.getUntrackedParameter<vector<int> >("maskedFEDs", listDefaults);
  seedCrys_ = iConfig.getUntrackedParameter<vector<int> >("seedCrys",vector<int>());

  vector<string> defaultMaskedEBs;
  defaultMaskedEBs.push_back("none");
  maskedEBs_ =  iConfig.getUntrackedParameter<vector<string> >("maskedEBs",defaultMaskedEBs);
  
  fedMap_ = new EcalFedMap();

  string title1 = "Jitter for all FEDs";
  string name1 = "JitterAllFEDs";
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

}


EcalMipGraphs::~EcalMipGraphs()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalMipGraphs::analyze(edm::Event const & iEvent, edm::EventSetup const & iSetup)
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
  }

  //We only want the 3x3's for this event...
  listEBChannels.clear();
  listEEChannels.clear();
  Handle<EcalRecHitCollection> EBhits;
  Handle<EcalRecHitCollection> EEhits;
  ESHandle<CaloTopology> caloTopo;
  iSetup.get<CaloTopologyRecord>().get(caloTopo);
  iEvent.getByLabel(EBRecHitCollection_, EBhits);
  iEvent.getByLabel(EERecHitCollection_, EEhits);
  // Now, retrieve the crystal digi from the event
  iEvent.getByLabel(EBDigis_, EBdigisHandle);
  iEvent.getByLabel(EEDigis_, EEdigisHandle);
  //debug
  //LogWarning("EcalMipGraphs") << "event " << ievt << " EBhits collection size " << EBhits->size();
  //LogWarning("EcalMipGraphs") << "event " << ievt << " EEhits collection size " << EEhits->size();

  selectHits(EBhits, ievt, caloTopo);
  selectHits(EEhits, ievt, caloTopo);
  
}


TGraph* EcalMipGraphs::selectDigi(DetId thisDet, int ievt)
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
      //LogWarning("EcalMipGraphs") << "Cannot find digi for ic:" << ic
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
      //LogWarning("EcalMipGraphs") << "Cannot find digi for ic:" << ic
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
    //edm::LogWarning("EcalMipGraphs")<< "itr->second(ampli)="<< itr->second;
    energy = (float) itr->second;
  }
  //else
  //edm::LogWarning("EcalMipGraphs") << "cry " << ic << "not found in ampMap";

  title+="_Energy"+floatToString(round(energy*1000));

  oneGraph->SetTitle(title.c_str());
  oneGraph->SetName(name.c_str());
  oneGraph->GetXaxis()->SetTitle("sample");
  oneGraph->GetYaxis()->SetTitle("ADC");
  return oneGraph;
}

void EcalMipGraphs::selectHits(Handle<EcalRecHitCollection> hits,
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
    int cryIndex = isBarrel ? ((EBDetId)det).hashedIndex() : ((EEDetId)det).hashedIndex();
    int ic = isBarrel ? ((EBDetId)det).ic() : getEEIndex(elecId);
    
    float ampli = hit.energy();

    vector<int>::iterator result;
    result = find(maskedFEDs_.begin(), maskedFEDs_.end(), FEDid);
    if(result != maskedFEDs_.end())
    {
      //LogWarning("EcalMipGraphs") << "skipping uncalRecHit for FED " << FEDid << " ; amplitude " << ampli;
      continue;
    }      
    result = find(maskedChannels_.begin(), maskedChannels_.end(), cryIndex);
    if  (result != maskedChannels_.end())
    {
      //LogWarning("EcalMipGraphs") << "skipping uncalRecHit for channel: " << cryIndex << " in fed: " << FEDid << " with amplitude " << ampli ;
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
      string name = "Event" + intToString(naiveEvtNum_) + "_ic" + intToString(ic)
        + "_FED" + intToString(FEDid);
      string title = "Digis";
      string seed = "ic" + intToString(ic) + "_FED" + intToString(FEDid);
      int freq=1;
      pair<map<string,int>::iterator,bool> pair = seedFrequencyMap_.insert(make_pair(seed,freq));
      if(!pair.second)
      {
        ++(pair.first->second);
      }
      
      TCanvas can(name.c_str(),title.c_str(),200,50,900,900);
      can.Divide(side_,side_);
      TGraph* myGraph;
      int canvasNum = 1;

      CaloNavigator<DetId> cursor = CaloNavigator<DetId>(det,caloTopo->getSubdetectorTopology(det));
      //Now put each graph in one by one
      for(int j=side_/2; j>=-side_/2; --j)
      {
        for(int i=-side_/2; i<=side_/2; ++i)
        {
          cursor.home();
          cursor.offsetBy(i,j);
          can.cd(canvasNum);
          myGraph = selectDigi(*cursor,ievt);
          myGraph->Draw("A*");
          canvasNum++;
        }
      }
      can.Write();
      names->push_back(name);
    }
    
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(timingHist==0)
    {
      initHists(FEDid);
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    if(ampli > minTimingAmp_)
    {
      timingHist->Fill(hit.time());
      allFedsTimingHist_->Fill(hit.time());
    }
  }
}

int EcalMipGraphs::getEEIndex(EcalElectronicsId elecId)
{
  int FEDid = 600+elecId.dccId();
  return 10000*FEDid+100*elecId.towerId()+5*(elecId.stripId()-1)+elecId.xtalId();
}

// insert the hist map into the map keyed by FED number
void EcalMipGraphs::initHists(int FED)
{
  using namespace std;
  
  string title1 = "Jitter for ";
  title1.append(fedMap_->getSliceFromFed(FED));
  string name1 = "JitterFED";
  name1.append(intToString(FED));
  TH1F* timingHist = fileService->make<TH1F>(name1.c_str(),title1.c_str(),150,-7,7);
  FEDsAndTimingHists_[FED] = timingHist;
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalMipGraphs::beginRun(edm::Run const &, edm::EventSetup const & c)
{
  edm::ESHandle< EcalElectronicsMapping > handle;
  c.get< EcalMappingRcd >().get(handle);
  ecalElectronicsMap_ = handle.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalMipGraphs::endJob()
{
  canvasNames_->Fill();

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
  LogWarning("EcalMipGraphs") << "Found seeds with frequency > 1: " << "\n\n" << frequencies;
  
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

  LogWarning("EcalMipGraphs") << "Masked channels are: " << channels;
  LogWarning("EcalMipGraphs") << "Masked FEDs are: " << feds << " and that is all!";
}


std::string EcalMipGraphs::intToString(int num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}

std::string EcalMipGraphs::floatToString(float num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}
