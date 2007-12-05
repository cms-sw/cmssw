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
// $Id: EcalURecHitHists.cc,v 1.7 2007/12/01 17:58:15 scooper Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "TFile.h"
#include "TH1F.h"
#include "TGraph.h"
#include "TNtuple.h"


//
// class declaration
//

class EcalURecHitHists : public edm::EDAnalyzer {
   public:
      explicit EcalURecHitHists(const edm::ParameterSet&);
      ~EcalURecHitHists();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      std::string intToString(int num);
      void initHists(int);

    // ----------member data ---------------------------

  edm::InputTag EcalUncalibratedRecHitCollection_;
  int runNum_;
  double histRangeMax_, histRangeMin_;
  std::string fileName_;

  std::vector<int> maskedChannels_;
  std::vector<int> maskedFEDs_;
  std::vector<std::string> maskedEBs_;
  std::map<int,TH1F*> FEDsAndHists_;

  TH1F* allFedsHist_;
  TFile* file;
  EcalFedMap* fedMap_;
};

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
  EcalUncalibratedRecHitCollection_ (iConfig.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection")),
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
  string name1 = "AllFeds";
  int numBins = (int)round(histRangeMax_-histRangeMin_)+1;
  allFedsHist_ = new TH1F(name1.c_str(),title1.c_str(),numBins,histRangeMin_,histRangeMax_);

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
EcalURecHitHists::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  int ievt = iEvent.id().event();
  auto_ptr<EcalElectronicsMapping> ecalElectronicsMap(new EcalElectronicsMapping);
  Handle<EcalUncalibratedRecHitCollection> hits;

  //TODO: improve try/catch behavior
  try
  {
    iEvent.getByLabel(EcalUncalibratedRecHitCollection_, hits);
    int neh = hits->size();
    LogDebug("EcalURecHitHists") << "event " << ievt << " hits collection size " << neh;
  }
  catch ( exception& ex)
  {
    LogWarning("EcalURecHitHists") << EcalUncalibratedRecHitCollection_ << " not available";
  }

  for (EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr)
  {
    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId ebDet = hit.id();
    //TODO: make it work for endcap FEDs also
    int ic = ebDet.ic();
    int hashedIndex = ebDet.hashedIndex();
    EcalElectronicsId elecId = ecalElectronicsMap->getElectronicsId(ebDet);
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
    TH1F* hist = FEDsAndHists_[FEDid];
    if(hist==0)
    {
      initHists(FEDid);
      hist = FEDsAndHists_[FEDid];
    }
    
    hist->Fill(ampli);
    allFedsHist_->Fill(ampli);
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
  string name1 = "FED";
  name1.append(intToString(FED));
  int numBins = (int)round(histRangeMax_-histRangeMin_)+1;
  TH1F* hist = new TH1F(name1.c_str(),title1.c_str(), numBins, histRangeMin_, histRangeMax_);
  FEDsAndHists_[FED] = hist;
  FEDsAndHists_[FED]->SetDirectory(0);
}

// ------------ method called once each job just before starting event loop  ------------
void 
EcalURecHitHists::beginJob(const edm::EventSetup&)
{
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
    root_file_.cd();
  }
  allFedsHist_->Write();
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

//define this as a plug-in
DEFINE_FWK_MODULE(EcalURecHitHists);
