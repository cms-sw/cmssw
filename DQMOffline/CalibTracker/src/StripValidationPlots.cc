// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2F.h"
#include "TH1D.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TTree.h"

#include <sstream>
#include <iostream>
#include <vector>

//
// class decleration
//

class StripValidationPlots : public edm::EDAnalyzer {
   public:
      explicit StripValidationPlots(const edm::ParameterSet&);
      ~StripValidationPlots();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  std::ostringstream oss;

  DQMStore* dqmStore_;

  MonitorElement* tmp;
  TProfile* tmp_prof;

  // Histograms
  std::vector<TH2F*> distanceVsStripNumber;
  std::vector<TProfile*> pfxDistanceVsStripNumber;
  std::vector<TH1F*> projXDistanceVsStripNumber;
  std::vector<TH1F*> projYDistanceVsStripNumber;

  std::vector<TH2F*> occupancyVsStripNumber;
  std::vector<TProfile*> pfxOccupancyVsStripNumber;
  std::vector<TH1F*> projYOccupancyVsStripNumber;
  std::vector<TH2F*> occupancyHotStripsVsStripNumber;
  std::vector<TProfile*> pfxOccupancyHotStripsVsStripNumber;
  std::vector<TH1F*> projYOccupancyHotStripsVsStripNumber;
  std::vector<TH2F*> occupancyGoodStripsVsStripNumber;
  std::vector<TProfile*> pfxOccupancyGoodStripsVsStripNumber;
  std::vector<TH1F*> projYOccupancyGoodStripsVsStripNumber;

  std::vector<TH2F*> poissonProbVsStripNumber;
  std::vector<TProfile*> pfxPoissonProbVsStripNumber;
  std::vector<TH1F*> projYPoissonProbVsStripNumber;
  std::vector<TH2F*> poissonProbHotStripsVsStripNumber;
  std::vector<TProfile*> pfxPoissonProbHotStripsVsStripNumber;
  std::vector<TH1F*> projYPoissonProbHotStripsVsStripNumber;
  std::vector<TH2F*> poissonProbGoodStripsVsStripNumber;
  std::vector<TProfile*> pfxPoissonProbGoodStripsVsStripNumber;
  std::vector<TH1F*> projYPoissonProbGoodStripsVsStripNumber;
   
  std::vector<TH2F*> nHitsVsStripNumber;
  std::vector<TProfile*> pfxNHitsVsStripNumber;
  std::vector<TH1F*> projXNHitsVsStripNumber;
  std::vector<TH1F*> projYNHitsVsStripNumber;
  std::vector<TH2F*> nHitsHotStripsVsStripNumber;
  std::vector<TProfile*> pfxNHitsHotStripsVsStripNumber;
  std::vector<TH1F*> projXNHitsHotStripsVsStripNumber;
  std::vector<TH1F*> projYNHitsHotStripsVsStripNumber;
  std::vector<TH2F*> nHitsGoodStripsVsStripNumber;
  std::vector<TProfile*> pfxNHitsGoodStripsVsStripNumber;
  std::vector<TH1F*> projXNHitsGoodStripsVsStripNumber;
  std::vector<TH1F*> projYNHitsGoodStripsVsStripNumber;
   
  std::vector<std::string> subDetName;

  std::string infilename;
  std::string outfilename;

  TFile* infile;
  TTree* intree;

  // Declaration of leaf types
  Int_t           DetRawId;
  Int_t           SubDetId;
  Int_t           Layer_Ring;
  Int_t           Disc;
  Int_t           IsBack;
  Int_t           IsExternalString;
  Int_t           IsZMinusSide;
  Int_t           RodStringPetal;
  Int_t           IsStereo;
  Int_t           ModulePosition;
  Int_t           NumberOfStrips;
  Int_t           StripNumber;
  Int_t           APVChannel;
  Float_t         StripGlobalPositionX;
  Float_t         StripGlobalPositionY;
  Float_t         StripGlobalPositionZ;
  Int_t           IsHot;
  Int_t           HotStripsPerAPV;
  Int_t           HotStripsPerModule;
  Double_t        StripOccupancy;
  Int_t           StripHits;
  Double_t        PoissonProb;


  Int_t savedDetRawId;
  Int_t savedNStrips;
  Int_t savedSubDetId;
  Int_t firstEntry;
  std::vector<unsigned int> vHotStripsInModule;
  unsigned int distance;
  unsigned int distanceR, distanceL;
  unsigned int nReadStrips[768][5];

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
StripValidationPlots::StripValidationPlots(const edm::ParameterSet& iConfig) : infilename(iConfig.getUntrackedParameter<std::string>("inputFilename","in.root")),
									       outfilename(iConfig.getUntrackedParameter<std::string>("outputFilename","out.root"))

{
   //now do what ever initialization is needed

}


StripValidationPlots::~StripValidationPlots()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
StripValidationPlots::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

}


// ------------ method called once each job just before starting event loop  ------------
void 
StripValidationPlots::beginJob()
{
  oss.str("");
  oss << 1; //runNumber

  dqmStore_ = edm::Service<DQMStore>().operator->();
  dqmStore_->setCurrentFolder("ChannelStatusPlots");

  for (int i=0; i<768; i++)
    {
      for (int j=0; j<5; j++)
	nReadStrips[i][j]=0;
    }

  // Initialize histograms
  subDetName.push_back(""); subDetName.push_back("TIB"); subDetName.push_back("TID"); subDetName.push_back("TOB"); subDetName.push_back("TEC");
  std::string histoName;
  std::string histoTitle;
  for(unsigned int i = 0; i < subDetName.size(); i++)
  {
    histoName = "distanceVsStripNumber" + subDetName[i];
    histoTitle = "Distance between hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 999, 0.5, 999.5);
    distanceVsStripNumber.push_back(tmp->getTH2F());

    histoName = "pfxDistanceVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxDistanceVsStripNumber.push_back(tmp->getTProfile());
    pfxDistanceVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxDistanceVsStripNumber[i]->GetYaxis()->SetTitle("Distance");

    histoName = "projXDistanceVsStripNumber" + subDetName[i];
    histoTitle = "Number of hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXDistanceVsStripNumber.push_back(tmp->getTH1F());
    projXDistanceVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    projXDistanceVsStripNumber[i]->GetYaxis()->SetTitle("N_{hot}");
    
    histoName = "projYDistanceVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of distance between hot strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 999, 0.5, 999.5);
    projYDistanceVsStripNumber.push_back(tmp->getTH1F());
    projYDistanceVsStripNumber[i]->GetXaxis()->SetTitle("Distance");
    projYDistanceVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "occupancyVsStripNumber" + subDetName[i];
    histoTitle = "Occupancy of strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -8.,0.);
    occupancyVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxOccupancyVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxOccupancyVsStripNumber.push_back(tmp->getTProfile());
    pfxOccupancyVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxOccupancyVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Occupancy)");
    
    histoName = "projYOccupancyVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of strip occupancy";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -8., 0.);
    projYOccupancyVsStripNumber.push_back(tmp->getTH1F());
    projYOccupancyVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
    projYOccupancyVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "occupancyHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Occupancy of hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -8., 0.);
    occupancyHotStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxOccupancyHotStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxOccupancyHotStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxOccupancyHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxOccupancyHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Occupancy)");
    
    histoName = "projYOccupancyHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of hot strip occupancy";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -8., 0.);
    projYOccupancyHotStripsVsStripNumber.push_back(tmp->getTH1F());
    projYOccupancyHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
    projYOccupancyHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "occupancyGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Occupancy of good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -8., 0.);
    occupancyGoodStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxOccupancyGoodStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxOccupancyGoodStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxOccupancyGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxOccupancyGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Occupancy)");
    
    histoName = "projYOccupancyGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of good strip occupancy";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -8., 0.);
    projYOccupancyGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    projYOccupancyGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
    projYOccupancyGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "poissonProbVsStripNumber" + subDetName[i];
    histoTitle = "Poisson probability of strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -18., 0.);
    poissonProbVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxPoissonProbVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxPoissonProbVsStripNumber.push_back(tmp->getTProfile());
    pfxPoissonProbVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxPoissonProbVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Probability)");
    
    histoName = "projYPoissonProbVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of strip Poisson probability";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -18., 0.);
    projYPoissonProbVsStripNumber.push_back(tmp->getTH1F());
    projYPoissonProbVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Probability)");
    projYPoissonProbVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "poissonProbHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Poisson probability of hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -18., 0.);
    poissonProbHotStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxPoissonProbHotStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxPoissonProbHotStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxPoissonProbHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxPoissonProbHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Probability)");
    
    histoName = "projYPoissonProbHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of hot strip Poisson probability";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -18., 0.);
    projYPoissonProbHotStripsVsStripNumber.push_back(tmp->getTH1F());
    projYPoissonProbHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Probability)");
    projYPoissonProbHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "poissonProbGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Poisson probability of good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 1000, -18., 0.);
    poissonProbGoodStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxPoissonProbGoodStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxPoissonProbGoodStripsVsStripNumber.push_back(tmp->getTProfile());
    pfxPoissonProbGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("Strip");
    pfxPoissonProbGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("log_{10}(Probability)");
    
    histoName = "projYPoissonProbGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of good strip Poisson probability";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -18., 0.);
    projYPoissonProbGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    projYPoissonProbGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("log_{10}(Probability)");
    projYPoissonProbGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");

    //
    histoName = "nHitsVsStripNumber" + subDetName[i];
    histoTitle = "NHits in strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 10000, -0.5, 9999.5);
    nHitsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxNHitsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxNHitsVsStripNumber.push_back(tmp->getTProfile());
    
    histoName = "projXNHitsVsStripNumber" + subDetName[i];
    histoTitle = "Cumulative nHits in strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXNHitsVsStripNumber.push_back(tmp->getTH1F());
    
    histoName = "projYNHitsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of nHits for all strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 10000, -0.5, 9999.5);
    projYNHitsVsStripNumber.push_back(tmp->getTH1F());
    projYNHitsVsStripNumber[i]->GetXaxis()->SetTitle("N_{hits}");
    projYNHitsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "nHitsHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "NHits in hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 10000, -0.5, 9999.5);
    nHitsHotStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxNHitsHotStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxNHitsHotStripsVsStripNumber.push_back(tmp->getTProfile());
    
    histoName = "projXNHitsHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Cumulative nHits in hot strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXNHitsHotStripsVsStripNumber.push_back(tmp->getTH1F());
    
    histoName = "projYNHitsHotStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of nHits for hot strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 10000, -0.5, 9999.5);
    projYNHitsHotStripsVsStripNumber.push_back(tmp->getTH1F());
    projYNHitsHotStripsVsStripNumber[i]->GetXaxis()->SetTitle("N_{hits}");
    projYNHitsHotStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
    
    //
    histoName = "nHitsGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "NHits in good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book2D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5, 10000, -0.5, 9999.5);
    nHitsGoodStripsVsStripNumber.push_back(tmp->getTH2F());
    
    histoName = "pfxNHitsGoodStripsVsStripNumber" + subDetName[i];
    tmp_prof = new TProfile(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    tmp = dqmStore_->bookProfile(histoName.c_str(), tmp_prof);
    pfxNHitsGoodStripsVsStripNumber.push_back(tmp->getTProfile());
    
    histoName = "projXNHitsGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Cumulative nHits in good strips vs. strip number";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 768, 0.5, 768.5);
    projXNHitsGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    
    histoName = "projYNHitsGoodStripsVsStripNumber" + subDetName[i];
    histoTitle = "Distribution of nHits for good strips";
    if(i!=0)
      histoTitle += " in " + subDetName[i];
    tmp = dqmStore_->book1D(histoName.c_str(), histoTitle.c_str(), 10000, -0.5, 9999.5);
    projYNHitsGoodStripsVsStripNumber.push_back(tmp->getTH1F());
    projYNHitsGoodStripsVsStripNumber[i]->GetXaxis()->SetTitle("N_{hits}");
    projYNHitsGoodStripsVsStripNumber[i]->GetYaxis()->SetTitle("N_{strips}");
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
StripValidationPlots::endJob() {

  infile = new TFile(infilename.c_str(),"READ");
  intree = (TTree*)infile->Get("stripOccupancy");

  intree->SetBranchAddress("DetRawId",             &DetRawId);
  intree->SetBranchAddress("SubDetId",             &SubDetId);
  intree->SetBranchAddress("Layer_Ring",           &Layer_Ring);
  intree->SetBranchAddress("Disc",                 &Disc);
  intree->SetBranchAddress("IsBack",               &IsBack);
  intree->SetBranchAddress("IsExternalString",     &IsExternalString);
  intree->SetBranchAddress("IsZMinusSide",         &IsZMinusSide);
  intree->SetBranchAddress("RodStringPetal",       &RodStringPetal);
  intree->SetBranchAddress("IsStereo",             &IsStereo);
  intree->SetBranchAddress("ModulePosition",       &ModulePosition);
  intree->SetBranchAddress("NumberOfStrips",       &NumberOfStrips);
  intree->SetBranchAddress("StripNumber",          &StripNumber);
  intree->SetBranchAddress("APVChannel",           &APVChannel);
  intree->SetBranchAddress("StripGlobalPositionX", &StripGlobalPositionX);
  intree->SetBranchAddress("StripGlobalPositionY", &StripGlobalPositionY);
  intree->SetBranchAddress("StripGlobalPositionZ", &StripGlobalPositionZ);
  intree->SetBranchAddress("IsHot",                &IsHot);
  intree->SetBranchAddress("HotStripsPerAPV",      &HotStripsPerAPV);
  intree->SetBranchAddress("HotStripsPerModule",   &HotStripsPerModule);
  intree->SetBranchAddress("StripOccupancy",       &StripOccupancy);
  intree->SetBranchAddress("StripHits",            &StripHits);
  intree->SetBranchAddress("PoissonProb",          &PoissonProb);

  for (int i=0; i<intree->GetEntries(); /* */)
    {
      intree->GetEntry(i);

      savedNStrips = NumberOfStrips;
      firstEntry = i;
      savedDetRawId = DetRawId;
      savedSubDetId = SubDetId;
      vHotStripsInModule.clear();

      for(int j = firstEntry; j < firstEntry + savedNStrips; j++)
	{
	  intree->GetEntry(j);

	  if(DetRawId != savedDetRawId)
	    {
	      std::cout << "ERROR! DetRawId changed but I expected to be on the same module!\n";
	      return;
	    }
       
	  double logStripOccupancy = log10(StripOccupancy);
	  double logPoissonProb = log10(fabs(PoissonProb));

	  (nReadStrips[StripNumber-1][0])++;
	  (nReadStrips[StripNumber-1][SubDetId-2])++;
	  //std::cout << "StripNumber = " << StripNumber << ", SubDetId = " << SubDetId << ", nReadStripsAll = " << nReadStrips[StripNumber-1][0] << ", nReadStripsSubDet = " << nReadStrips[StripNumber-1][SubDetId-2] << std::endl;
	  occupancyVsStripNumber[0]->Fill(StripNumber,logStripOccupancy);
	  occupancyVsStripNumber[SubDetId-2]->Fill(StripNumber,logStripOccupancy);
	  poissonProbVsStripNumber[0]->Fill(StripNumber,logPoissonProb);
	  poissonProbVsStripNumber[SubDetId-2]->Fill(StripNumber,logPoissonProb);
	  nHitsVsStripNumber[0]->Fill(StripNumber,StripHits);
	  nHitsVsStripNumber[SubDetId-2]->Fill(StripNumber,StripHits);
       
	  if(IsHot)
	    {
	      vHotStripsInModule.push_back(StripNumber);
	      //          std::cout << "detId = " << DetRawId << ", StripN = " << StripNumber << ", StripOccupancy = " << StripOccupancy << " PoissonProb = " << PoissonProb << std::endl ;
	      occupancyHotStripsVsStripNumber[0]->Fill(StripNumber,logStripOccupancy);
	      occupancyHotStripsVsStripNumber[SubDetId-2]->Fill(StripNumber,logStripOccupancy);
	      poissonProbHotStripsVsStripNumber[0]->Fill(StripNumber,logPoissonProb);
	      poissonProbHotStripsVsStripNumber[SubDetId-2]->Fill(StripNumber,logPoissonProb);
	      nHitsHotStripsVsStripNumber[0]->Fill(StripNumber,StripHits);
	      nHitsHotStripsVsStripNumber[SubDetId-2]->Fill(StripNumber,StripHits);
	    }
	  else
	    {
	      occupancyGoodStripsVsStripNumber[0]->Fill(StripNumber,logStripOccupancy);
	      occupancyGoodStripsVsStripNumber[SubDetId-2]->Fill(StripNumber,logStripOccupancy);
	      poissonProbGoodStripsVsStripNumber[0]->Fill(StripNumber,logPoissonProb);
	      poissonProbGoodStripsVsStripNumber[SubDetId-2]->Fill(StripNumber,logPoissonProb);
	      nHitsGoodStripsVsStripNumber[0]->Fill(StripNumber,StripHits);
	      nHitsGoodStripsVsStripNumber[SubDetId-2]->Fill(StripNumber,StripHits);
	    }
	}
     
      if(vHotStripsInModule.size()==1)
	{
	  distance = 999;
	  distanceVsStripNumber[0]->Fill(vHotStripsInModule[0], distance);
	  distanceVsStripNumber[savedSubDetId-2]->Fill(vHotStripsInModule[0], distance);
	}
      else if(vHotStripsInModule.size()>1)
	{
	  for(unsigned int iVec = 0; iVec != vHotStripsInModule.size(); iVec++)
	    {
	      if(iVec==0)
		distance = vHotStripsInModule[1] - vHotStripsInModule[0];
	      else if(iVec==vHotStripsInModule.size()-1)
		{
		  distance = vHotStripsInModule[vHotStripsInModule.size()-1] - vHotStripsInModule[vHotStripsInModule.size() -2];
		}
	      else if(vHotStripsInModule.size()>2)
		{
		  distanceR = vHotStripsInModule[iVec + 1] -  vHotStripsInModule[iVec];
		  distanceL = vHotStripsInModule[iVec] - vHotStripsInModule[iVec - 1];
		  distance = distanceL>distanceR?distanceR:distanceL;
		}
	      else
		{
		  std::cout << "ERROR! distance is never computed!!!\n";
		}
	      //          std::cout << "detId = " << savedDetRawId << ", StripN = " << vHotStripsInModule[iVec] << ", distance = " << distance << std::endl;
	      distanceVsStripNumber[0]->Fill(vHotStripsInModule[iVec], distance);
	      distanceVsStripNumber[savedSubDetId-2]->Fill(vHotStripsInModule[iVec], distance);
	    }
	}
     
      i+=savedNStrips;

    }


  std::string histoName;
  std::string histoTitle;

  for(unsigned int i = 0; i < subDetName.size(); i++)
    {
      projYDistanceVsStripNumber[i]->Add((TH1F*)distanceVsStripNumber[i]->ProjectionY());
      pfxDistanceVsStripNumber[i]->Add(distanceVsStripNumber[i]->ProfileX(pfxDistanceVsStripNumber[i]->GetName(),1,998));
      projYNHitsVsStripNumber[i]->Add(nHitsVsStripNumber[i]->ProjectionY());
      projYNHitsGoodStripsVsStripNumber[i]->Add(nHitsGoodStripsVsStripNumber[i]->ProjectionY());
      projYNHitsHotStripsVsStripNumber[i]->Add(nHitsHotStripsVsStripNumber[i]->ProjectionY());
      projYOccupancyVsStripNumber[i]->Add(occupancyVsStripNumber[i]->ProjectionY());
      projYOccupancyGoodStripsVsStripNumber[i]->Add(occupancyGoodStripsVsStripNumber[i]->ProjectionY());
      projYOccupancyHotStripsVsStripNumber[i]->Add(occupancyHotStripsVsStripNumber[i]->ProjectionY());
      pfxOccupancyVsStripNumber[i]->Add(occupancyVsStripNumber[i]->ProfileX(pfxOccupancyVsStripNumber[i]->GetName(),-8.,0.));
      pfxOccupancyGoodStripsVsStripNumber[i]->Add(occupancyGoodStripsVsStripNumber[i]->ProfileX(pfxOccupancyGoodStripsVsStripNumber[i]->GetName(),-8.,0.));
      pfxOccupancyHotStripsVsStripNumber[i]->Add(occupancyHotStripsVsStripNumber[i]->ProfileX(pfxOccupancyHotStripsVsStripNumber[i]->GetName(),-8.,0.));
      projYPoissonProbVsStripNumber[i]->Add(poissonProbVsStripNumber[i]->ProjectionY());
      projYPoissonProbGoodStripsVsStripNumber[i]->Add(poissonProbGoodStripsVsStripNumber[i]->ProjectionY());
      projYPoissonProbHotStripsVsStripNumber[i]->Add(poissonProbHotStripsVsStripNumber[i]->ProjectionY());
      pfxPoissonProbVsStripNumber[i]->Add(poissonProbVsStripNumber[i]->ProfileX(pfxPoissonProbVsStripNumber[i]->GetName(),-18., 0.));
      pfxPoissonProbGoodStripsVsStripNumber[i]->Add(poissonProbGoodStripsVsStripNumber[i]->ProfileX(pfxPoissonProbGoodStripsVsStripNumber[i]->GetName(),-18., 0.));
      pfxPoissonProbHotStripsVsStripNumber[i]->Add(poissonProbHotStripsVsStripNumber[i]->ProfileX(pfxPoissonProbHotStripsVsStripNumber[i]->GetName(),-18., 0.));
      projXDistanceVsStripNumber[i]->Add(distanceVsStripNumber[i]->ProjectionX(projXDistanceVsStripNumber[i]->GetName(),1,998));

    }

  dqmStore_->cd();
  dqmStore_->save(outfilename.c_str(),"ChannelStatusPlots");

}

//define this as a plug-in
DEFINE_FWK_MODULE(StripValidationPlots);
