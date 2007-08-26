// -*- C++ -*-
//
// Package:    RCTMonitor
// Class:      RCTMonitor
// 
/**\class RCTMonitor RCTMonitor.cc Analysis/RCTMonitor/src/RCTMonitor.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sridhara Dasu
//         Created:  Sat Aug 25 11:15:13 CEST 2007
// $Id: RCTMonitor.cc,v 1.1 2007/08/25 15:22:29 dasu Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <iomanip>

#include <vector>
using std::vector;

#include <string>
using std::string;

#include "TROOT.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"

//
// class declaration
//

class RCTMonitor : public edm::EDAnalyzer {
   public:
      explicit RCTMonitor(const edm::ParameterSet&);
      ~RCTMonitor();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  int nEvents;

  std::string outputFileName;

  int regionETCut;
  int emCandRankCut;

  TFile *outFile;

  vector<TH1F *> plots1D;
  vector<TH2F *> plots2D;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

TH1F* define1DHistogram(const char* name, const char* title,
			const char* xTitle, const char* yTitle,
			Int_t color, Int_t type,
			Int_t nBins, Float_t low, Float_t high)
{
  TH1F* histogram = new TH1F(name, title, nBins, low, high);
  histogram->GetXaxis()->SetTitle(xTitle);
  histogram->GetYaxis()->SetTitle(yTitle);
  histogram->SetLineColor(color);
  histogram->SetMarkerColor(color);
  histogram->SetMarkerStyle(type);
  return histogram;
}

TH2F* define2DHistogram(const char* name, const char* title,
			const char* xTitle, const char* yTitle, const char* zTitle,
			Int_t color, Int_t type,
			Int_t nxBins, Float_t xLow, Float_t xHigh,
			Int_t nyBins, Float_t yLow, Float_t yHigh)
{
  TH2F* histogram = new TH2F(name, title, nxBins, xLow, xHigh, nyBins, yLow, yHigh);
  histogram->GetXaxis()->SetTitle(xTitle);
  histogram->GetYaxis()->SetTitle(yTitle);
  histogram->GetZaxis()->SetTitle(zTitle);
  histogram->SetLineColor(color);
  histogram->SetMarkerColor(color);
  histogram->SetMarkerStyle(type);
  histogram->SetOption("TEXTBOX");
  histogram->SetShowProjectionX(true);
  histogram->SetShowProjectionY(true);
  return histogram;
}

//
// constructors and destructor
//
RCTMonitor::RCTMonitor(const edm::ParameterSet& iConfig) :
  nEvents(0),
  outputFileName(iConfig.getParameter<std::string>("outputFile")),
  regionETCut(iConfig.getParameter<int>("regionETCut")),
  emCandRankCut(iConfig.getParameter<int>("emCandRankCut"))
{
}


RCTMonitor::~RCTMonitor()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RCTMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  Handle<L1CaloEmCollection> rctEmCands;
  Handle<L1CaloRegionCollection> rctRegions;
  iEvent.getByType(rctEmCands);
  iEvent.getByType(rctRegions);
  L1CaloEmCollection::const_iterator em;
  L1CaloRegionCollection::const_iterator rgn;
  for (rgn=rctRegions->begin(); rgn!=rctRegions->end(); rgn++){
    double_t eta = rgn->gctEta();
    double_t phi = rgn->gctPhi();
    double_t et = rgn->et();
    plots2D[0]->Fill(eta, phi, et);
    if(et > regionETCut) plots2D[1]->Fill(eta, phi);
  }  
  for (em=rctEmCands->begin(); em!=rctEmCands->end(); em++){
    double_t eta = em->regionId().ieta();
    double_t phi = em->regionId().iphi();
    double_t rank = em->rank();
    bool isolated = em->isolated();
    if(isolated)
      {
	plots2D[2]->Fill(eta, phi, rank);
	if(rank > emCandRankCut) plots2D[3]->Fill(eta, phi);
      }
    else
      {
	plots2D[4]->Fill(eta, phi, rank);
	if(rank > emCandRankCut) plots2D[5]->Fill(eta, phi);
      }
  }  
}

// ------------ method called once each job just before starting event loop  ------------
void 
RCTMonitor::beginJob(const edm::EventSetup&)
{
  outFile = new TFile(outputFileName.c_str(),"RECREATE", outputFileName.c_str());
  outFile->cd();
  plots2D.push_back(define2DHistogram("RCTRegionET", 
				      "RCTRegion ET vs (iEta, iPhi)",
				      "iEta", "iPhi", "Region ET",
				      kRed, kFullSquare,
				      25, 0.0, 25.0, 20, 0.0, 20.0));
  plots2D.push_back(define2DHistogram("RCTRegionOcc", 
				      "RCTRegion Occupancy vs (iEta, iPhi)",
				      "iEta", "iPhi", "Region Occupancy",
				      kBlue, kFullCircle,
				      25, 0.0, 25.0, 20, 0.0, 20.0));
  plots2D.push_back(define2DHistogram("RCTIsoEMRank", 
				      "RCTIsoEM Rank vs (iEta, iPhi)",
				      "iEta", "iPhi", "RCTIsoEM Rank",
				      kRed, kFullSquare,
				      25, 0.0, 25.0, 20, 0.0, 20.0));
  plots2D.push_back(define2DHistogram("RCTIsoEMOcc", 
				      "RCTIsoEM Occupancy vs (iEta, iPhi)",
				      "iEta", "iPhi", "RCTIsoEM Occupancy",
				      kBlue, kFullCircle,
				      25, 0.0, 25.0, 20, 0.0, 20.0));
  plots2D.push_back(define2DHistogram("RCTnonIsoEMRank", 
				      "RCTnonIsoEM Rank vs (iEta, iPhi)",
				      "iEta", "iPhi", "RCTnonIsoEM Rank",
				      kMagenta, kFullTriangleUp,
				      25, 0.0, 25.0, 20, 0.0, 20.0));
  plots2D.push_back(define2DHistogram("RCTnonIsoEMOcc", 
				      "RCTnonIsoEM Occupancy vs (iEta, iPhi)",
				      "iEta", "iPhi", "RCTnonIsoEM Occupancy",
				      kBlack, kFullTriangleDown,
				      25, 0.0, 25.0, 20, 0.0, 20.0));
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RCTMonitor::endJob() 
{
  for(int i = 0; i < 6; i++)
    {
      plots2D[i]->ProjectionX();
      plots2D[i]->ProjectionY();
    }
  outFile->cd();
  outFile->Write();
  outFile->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(RCTMonitor);
