// -*- C++ -*-
//
// Package:   EcalURecHitHists 
// Class:     EcalURecHitHists 
// 
/**\class EcalURecHitHists EcalURecHitHists.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seth COOPER
//         Created:  Th Nov 22 5:46:22 CEST 2007
// $Id: EcalURecHitHists.cc,v 1.5 2010/01/04 15:07:40 ferriff Exp $
//
//

#include "CaloOnlineTools/EcalTools/plugins/EcalURecHitHists.h"

using namespace cms;
using namespace edm;
using namespace std;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalURecHitHists::EcalURecHitHists(const edm::ParameterSet& iConfig) :
  EBUncalibratedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EBUncalibratedRecHitCollection")),
  EEUncalibratedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EEUncalibratedRecHitCollection")),
  runNum_(-1),
  histRangeMax_ (iConfig.getUntrackedParameter<double>("histogramMaxRange",200.0)),
  histRangeMin_ (iConfig.getUntrackedParameter<double>("histogramMinRange",-10.0)),
  fileName_ (iConfig.getUntrackedParameter<std::string>("fileName", std::string("ecalURechHitHists")))
{
  vector<int> listDefaults;
  listDefaults.push_back(-1);
  
  maskedChannels_ = iConfig.getUntrackedParameter<vector<int> >("maskedChannels", listDefaults);
  maskedFEDs_ = iConfig.getUntrackedParameter<vector<int> >("maskedFEDs", listDefaults);

  vector<string> defaultMaskedEBs;
  defaultMaskedEBs.push_back("none");
  maskedEBs_ =  iConfig.getUntrackedParameter<vector<string> >("maskedEBs",defaultMaskedEBs);
  
  fedMap_ = new EcalFedMap();
  string title1 = "Uncalib Rec Hits (ADC counts)";
  string name1 = "URecHitsAllFEDs";
  int numBins = (int)round(histRangeMax_-histRangeMin_)+1;
  allFedsHist_ = new TH1F(name1.c_str(),title1.c_str(),numBins,histRangeMin_,histRangeMax_);
  title1 = "Jitter for all FEDs";
  name1 = "JitterAllFEDs";
  allFedsTimingHist_ = new TH1F(name1.c_str(),title1.c_str(),14,-7,7);

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
}


EcalURecHitHists::~EcalURecHitHists()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
EcalURecHitHists::analyze(edm::Event const & iEvent, edm::EventSetup const & iSetup)
{
  int ievt = iEvent.id().event();
  Handle<EcalUncalibratedRecHitCollection> EBhits;
  Handle<EcalUncalibratedRecHitCollection> EEhits;

  iEvent.getByLabel(EBUncalibratedRecHitCollection_, EBhits);
  LogDebug("EcalURecHitHists") << "event " << ievt << " hits collection size " << EBhits->size();

  iEvent.getByLabel(EEUncalibratedRecHitCollection_, EEhits);
  LogDebug("EcalURecHitHists") << "event " << ievt << " hits collection size " << EEhits->size();

  for (EcalUncalibratedRecHitCollection::const_iterator hitItr = EBhits->begin(); hitItr != EBhits->end(); ++hitItr)
  {
    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId ebDet = hit.id();
    int ic = ebDet.ic();
    int hashedIndex = ebDet.hashedIndex();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(ebDet);
    int FEDid = 600+elecId.dccId();
    float ampli = hit.amplitude();

    vector<int>::iterator result;
    result = find(maskedFEDs_.begin(), maskedFEDs_.end(), FEDid);
    if(result != maskedFEDs_.end())
    {
      LogWarning("EcalURecHitHists") << "skipping uncalRecHit for FED " << FEDid << " ; amplitude " << ampli;
      continue;
    }      

    result = find(maskedChannels_.begin(), maskedChannels_.end(), hashedIndex);
    if  (result != maskedChannels_.end())
    {
      LogWarning("EcalURecHitHists") << "skipping uncalRecHit for channel: " << ic << " with amplitude " << ampli ;
      continue;
    }      

    // fill the proper hist
    TH1F* uRecHist = FEDsAndHists_[FEDid];
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(uRecHist==0)
    {
      initHists(FEDid);
      uRecHist = FEDsAndHists_[FEDid];
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    
    uRecHist->Fill(ampli);
    allFedsHist_->Fill(ampli);
    timingHist->Fill(hit.jitter());
    allFedsTimingHist_->Fill(hit.jitter());
  }
  
  // Again for the endcap
  for (EcalUncalibratedRecHitCollection::const_iterator hitItr = EEhits->begin(); hitItr != EEhits->end(); ++hitItr)
  {
    EcalUncalibratedRecHit hit = (*hitItr);
    EEDetId eeDet = hit.id();
    int ic = eeDet.ic();
    int hashedIndex = eeDet.hashedIndex();
    EcalElectronicsId elecId = ecalElectronicsMap_->getElectronicsId(eeDet);
    int FEDid = 600+elecId.dccId();
    float ampli = hit.amplitude();

    vector<int>::iterator result;
    result = find(maskedFEDs_.begin(), maskedFEDs_.end(), FEDid);
    if(result != maskedFEDs_.end())
    {
      LogWarning("EcalURecHitHists") << "skipping uncalRecHit for FED " << FEDid << " ; amplitude " << ampli;
      continue;
    }      

    result = find(maskedChannels_.begin(), maskedChannels_.end(), hashedIndex);
    if  (result != maskedChannels_.end())
    {
      LogWarning("EcalURecHitHists") << "skipping uncalRecHit for channel: " << ic << " with amplitude " << ampli ;
      continue;
    }      

    // fill the proper hist
    TH1F* uRecHist = FEDsAndHists_[FEDid];
    TH1F* timingHist = FEDsAndTimingHists_[FEDid];
    if(uRecHist==0)
    {
      initHists(FEDid);
      uRecHist = FEDsAndHists_[FEDid];
      timingHist = FEDsAndTimingHists_[FEDid];
    }
    
    uRecHist->Fill(ampli);
    allFedsHist_->Fill(ampli);
    timingHist->Fill(hit.jitter());
    allFedsTimingHist_->Fill(hit.jitter());
  }

  if(runNum_==-1)
  {
    runNum_ = iEvent.id().run();
  }
}


// insert the hist map into the map keyed by FED number
void EcalURecHitHists::initHists(int FED)
{
  using namespace std;
  
  string FEDid = intToString(FED);
  string title1 = "Uncalib Rec Hits (ADC counts) for ";
  title1.append(fedMap_->getSliceFromFed(FED));
  string name1 = "URecHitsFED";
  name1.append(intToString(FED));
  int numBins = (int)round(histRangeMax_-histRangeMin_)+1;
  TH1F* hist = new TH1F(name1.c_str(),title1.c_str(), numBins, histRangeMin_, histRangeMax_);
  FEDsAndHists_[FED] = hist;
  FEDsAndHists_[FED]->SetDirectory(0);
  
  title1 = "Jitter for ";
  title1.append(fedMap_->getSliceFromFed(FED));
  name1 = "JitterFED";
  name1.append(intToString(FED));
  TH1F* timingHist = new TH1F(name1.c_str(),title1.c_str(),14,-7,7);
  FEDsAndTimingHists_[FED] = timingHist;
  FEDsAndTimingHists_[FED]->SetDirectory(0);
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalURecHitHists::beginRun(edm::Run const &, edm::EventSetup const & c)
{
  edm::ESHandle<EcalElectronicsMapping> elecHandle;
  c.get<EcalMappingRcd>().get(elecHandle);
  ecalElectronicsMap_ = elecHandle.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EcalURecHitHists::endJob()
{
  using namespace std;
  fileName_ += "-"+intToString(runNum_)+".graph.root";

  TFile root_file_(fileName_.c_str() , "RECREATE");

  for(map<int,TH1F*>::const_iterator itr = FEDsAndHists_.begin();
      itr != FEDsAndHists_.end(); ++itr)
  {
    string dir = fedMap_->getSliceFromFed(itr->first);
    TDirectory* FEDdir = gDirectory->mkdir(dir.c_str());
    FEDdir->cd();

    TH1F* hist = itr->second;
    if(hist!=0)
      hist->Write();
    else
    {
      cerr << "EcalPedHists: Error: This shouldn't happen!" << endl;
    }
    // Write out timing hist
    hist = FEDsAndTimingHists_[itr->first];
    if(hist!=0)
      hist->Write();
    else
    {
      cerr << "EcalPedHists: Error: This shouldn't happen!" << endl;
    }
    root_file_.cd();
  }
  allFedsHist_->Write();
  allFedsTimingHist_->Write();
  root_file_.Close();

  std::string channels;
  for(std::vector<int>::const_iterator itr = maskedChannels_.begin();
      itr != maskedChannels_.end(); ++itr)
  {
    channels+=intToString(*itr);
    channels+=",";
  }
  
  LogWarning("EcalMipGraphs") << "Masked channels are: " << channels << " and that is all!";
}


std::string EcalURecHitHists::intToString(int num)
{
    using namespace std;
    ostringstream myStream;
    myStream << num << flush;
    return(myStream.str()); //returns the string form of the stringstream object
}

