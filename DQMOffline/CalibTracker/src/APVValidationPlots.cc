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

class APVValidationPlots : public edm::EDAnalyzer {
   public:
      explicit APVValidationPlots(const edm::ParameterSet&);
      ~APVValidationPlots();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  std::ostringstream oss;

  DQMStore* dqmStore;

  MonitorElement* tmp;

  // Histograms
  // indexes in these arrays are [SubDetId-2][LayerN]
  // histograms for [SubDetId-2][0] are global for the subdetector
  // histogram for [0][0] is global for the tracker
  
  TH2F* medianVsAbsoluteOccupancy[5][10];
  TH1F* medianOccupancy[5][10];
  TH1F* absoluteOccupancy[5][10];

  std::vector<std::string> subDetName;
  std::vector<unsigned int> nLayers;
  std::vector<std::string> layerName;

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
   Float_t         APVGlobalPositionX;
   Float_t         APVGlobalPositionY;
   Float_t         APVGlobalPositionZ;
   Int_t           APVNumber;
   Int_t           APVAbsoluteOccupancy;
   Double_t        APVMedianOccupancy;



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
APVValidationPlots::APVValidationPlots(const edm::ParameterSet& iConfig) : infilename(iConfig.getUntrackedParameter<std::string>("inputFilename","in.root")),
									   outfilename(iConfig.getUntrackedParameter<std::string>("outputFilename","out.root"))

{
   //now do what ever initialization is needed

}


APVValidationPlots::~APVValidationPlots()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
APVValidationPlots::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

}


// ------------ method called once each job just before starting event loop  ------------
void 
APVValidationPlots::beginJob()
{
  oss.str("");
  oss << 1; //runNumber

  dqmStore = edm::Service<DQMStore>().operator->();
  dqmStore->setCurrentFolder("ChannelStatusPlots");

  // Initialize histograms
  subDetName.push_back(""); subDetName.push_back("TIB"); subDetName.push_back("TID"); subDetName.push_back("TOB"); subDetName.push_back("TEC");
  nLayers.push_back(0); nLayers.push_back(4); nLayers.push_back(3); nLayers.push_back(6); nLayers.push_back(9);
  layerName.push_back(""); layerName.push_back("Layer"); layerName.push_back("Disk"); layerName.push_back("Layer"); layerName.push_back("Disk");
  
  std::string histoName;
  std::string histoTitle;
  for(unsigned int i = 0; i < subDetName.size(); i++)
  {
    for(unsigned int j = 0; j <= nLayers[i]; j++)
    {
      histoName = "medianVsAbsoluteOccupancy" + subDetName[i];
      if(j!=0)
      {
        oss.str("");
        oss << j;
        histoName += layerName[i] + oss.str();
      }
      histoTitle = "Median APV occupancy vs. absolute APV occupancy";
      if(i!=0)
        histoTitle += " in " + subDetName[i];
      if(j!=0)
      {
        histoTitle += " " + layerName[i] + " " + oss.str();
      }
      tmp = dqmStore->book2D(histoName.c_str(), histoTitle.c_str(), 1000, 0., 6., 1000, -1., 3.);
      medianVsAbsoluteOccupancy[i][j] = tmp->getTH2F();
      medianVsAbsoluteOccupancy[i][j]->Rebin2D(10,10);
      medianVsAbsoluteOccupancy[i][j]->GetXaxis()->SetTitle("log_{10}(Abs. Occupancy)");
      medianVsAbsoluteOccupancy[i][j]->GetYaxis()->SetTitle("log_{10}(Median Occupancy)");
      //
      histoName = "medianOccupancy" + subDetName[i];
      if(j!=0)
      {
        oss.str("");
        oss << j;
        histoName += layerName[i] + oss.str();
      }
      histoTitle = "Median APV occupancy";
      if(i!=0)
        histoTitle += " in " + subDetName[i];
      if(j!=0)
      {
        histoTitle += " " + layerName[i] + " " + oss.str();
      }
      tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, -1., 3.);
      medianOccupancy[i][j] = tmp->getTH1F();
      medianOccupancy[i][j]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
      medianOccupancy[i][j]->GetYaxis()->SetTitle("APVs");
      //
      histoName = "absoluteOccupancy" + subDetName[i];
      if(j!=0)
      {
        oss.str("");
        oss << j;
        histoName += layerName[i] + oss.str();
      }
      histoTitle = "Absolute APV occupancy";
      if(i!=0)
        histoTitle += " in " + subDetName[i];
      if(j!=0)
      {
        histoTitle += " " + layerName[i] + " " + oss.str();
      }
      tmp = dqmStore->book1D(histoName.c_str(), histoTitle.c_str(), 1000, 0., 6.);
      absoluteOccupancy[i][j] = tmp->getTH1F();
      absoluteOccupancy[i][j]->GetXaxis()->SetTitle("log_{10}(Occupancy)");
      absoluteOccupancy[i][j]->GetYaxis()->SetTitle("APVs");
    }
  }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
APVValidationPlots::endJob() {

  infile = new TFile(infilename.c_str(),"READ");
  intree = (TTree*)infile->Get("moduleOccupancy");

  intree->SetBranchAddress("DetRawId",             &DetRawId);
  intree->SetBranchAddress("SubDetId",             &SubDetId);
  intree->SetBranchAddress("Layer_Ring",           &Layer_Ring);
  intree->SetBranchAddress("Disc",                 &Disc);
  intree->SetBranchAddress("IsBack",               &IsBack);
  intree->SetBranchAddress("IsExternalString",     &IsExternalString);
  intree->SetBranchAddress("IsZMinusSide",         &IsZMinusSide);
  intree->SetBranchAddress("RodStringPetal",       &RodStringPetal);
  intree->SetBranchAddress("IsStereo",             &IsStereo);
  intree->SetBranchAddress("ModuleNumber",         &ModulePosition);
  intree->SetBranchAddress("NumberOfStrips",       &NumberOfStrips);
  intree->SetBranchAddress("APVGlobalPositionX",   &APVGlobalPositionX);
  intree->SetBranchAddress("APVGlobalPositionY",   &APVGlobalPositionY);
  intree->SetBranchAddress("APVGlobalPositionZ",   &APVGlobalPositionZ);
  intree->SetBranchAddress("APVNumber",            &APVNumber);
  intree->SetBranchAddress("APVAbsoluteOccupancy", &APVAbsoluteOccupancy);
  intree->SetBranchAddress("APVMedianOccupancy",   &APVMedianOccupancy);

  for (int i=0; i<intree->GetEntries(); i++)
    {
      intree->GetEntry(i);

      double logMedianOccupancy = -1;
      double logAbsoluteOccupancy = -1;

      if (APVMedianOccupancy>0) logMedianOccupancy = log10(APVMedianOccupancy);
      if (APVAbsoluteOccupancy>0) logAbsoluteOccupancy = log10(APVAbsoluteOccupancy);

      // The layer/disk information is stored in Layer_Ring for TIB/TOB and in Disc for TID/TEC
      unsigned int layer = 0;
      if(SubDetId==3 || SubDetId==5)
        layer=Layer_Ring;
      else
        layer=Disc;

      // Fill histograms for all the tracker
      medianVsAbsoluteOccupancy[0][0]->Fill(logAbsoluteOccupancy,logMedianOccupancy);
      medianOccupancy[0][0]->Fill(logMedianOccupancy);
      absoluteOccupancy[0][0]->Fill(logAbsoluteOccupancy);
      // Fill summary histograms for each subdetector
      medianVsAbsoluteOccupancy[SubDetId-2][0]->Fill(logAbsoluteOccupancy,logMedianOccupancy);
      medianOccupancy[SubDetId-2][0]->Fill(logMedianOccupancy);
      absoluteOccupancy[SubDetId-2][0]->Fill(logAbsoluteOccupancy);
      // Fill histograms for each layer/disk
      medianVsAbsoluteOccupancy[SubDetId-2][layer]->Fill(logAbsoluteOccupancy,logMedianOccupancy);
      medianOccupancy[SubDetId-2][layer]->Fill(logMedianOccupancy);
      absoluteOccupancy[SubDetId-2][layer]->Fill(logAbsoluteOccupancy);
    }

  dqmStore->cd();
  dqmStore->save(outfilename.c_str(),"ChannelStatusPlots");

}

//define this as a plug-in
DEFINE_FWK_MODULE(APVValidationPlots);
