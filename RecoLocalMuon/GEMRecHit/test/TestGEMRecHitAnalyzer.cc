// -*- C++ -*-
//
// Package:    TestGEMRecHitAnalyzer
// Class:      TestGEMRecHitAnalyzer
// 
/**\class TestGEMRecHitAnalyzer TestGEMRecHitAnalyzer.cc MyAnalyzers/TestGEMRecHitAnalyzer/src/TestGEMRecHitAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Piet Verwilligen,161 R-006,+41227676292,
//         Created:  Wed Oct 24 17:28:30 CEST 2012
//
//


// system include files
#include <memory>
#include <fstream>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>


// root include files
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TF1.h"
#include "THStack.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TDirectoryFile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include <DataFormats/GEMDigi/interface/GEMDigiCollection.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
 
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <Geometry/GEMGeometry/interface/GEMEtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include "DataFormats/Provenance/interface/Timestamp.h"

#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>


#include <DataFormats/GEMDigi/interface/GEMDigiCollection.h>
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>



//
// class declaration
//

class TestGEMRecHitAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TestGEMRecHitAnalyzer(const edm::ParameterSet&);
      ~TestGEMRecHitAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
  edm::ESHandle<GEMGeometry> gemGeom;
  edm::EDGetTokenT<GEMRecHitCollection> GEMRecHit_Token;

  std::string rootFileName;
  std::unique_ptr<TFile> outputfile;
  std::unique_ptr<TH1F> BX_RecHits_GE1in_Plus,  BX_RecHits_GE1out_Plus,  BX_RecHits_GE1in_Minus,  BX_RecHits_GE1out_Minus;
  std::unique_ptr<TH1F> ST_RecHits_GE1in_Plus, ST_RecHits_GE1out_Plus, ST_RecHits_GE1in_Minus, ST_RecHits_GE1out_Minus;
  std::unique_ptr<TH1F> CL_RecHits_GE1in_Plus, CL_RecHits_GE1out_Plus, CL_RecHits_GE1in_Minus, CL_RecHits_GE1out_Minus;
  std::unique_ptr<TCanvas> BX_RecHits_GE1, ST_RecHits_GE1, CL_RecHits_GE1;

  // GEM Station 1 = GE1/1
  std::unique_ptr<TGraph>  GE1in_Plus_XY_All, GE1in_Minus_XY_All, GE1out_Plus_XY_All, GE1out_Minus_XY_All, GE1out_Plus_YZ_All, GE1out_Minus_YZ_All, GE1in_Plus_YZ_All, GE1in_Minus_YZ_All, GE1_Plus_YZ_All, GE1_Minus_YZ_All;
  std::unique_ptr<TCanvas> Canvas_GE1_Plus_XY, Canvas_GE1_Minus_XY, Canvas_GE1_Plus_YZ, Canvas_GE1_Minus_YZ;
  std::vector<double> x_n1i, y_n1i, z_n1i, r_n1i, x_n1o, y_n1o, z_n1o, r_n1o; // XYZR GE1 Minus in and out 
  std::vector<double> x_p1i, y_p1i, z_p1i, r_p1i, x_p1o, y_p1o, z_p1o, r_p1o; // XYZR GE1 Plus in and out
  std::vector<double> x_n1, y_n1, z_n1, r_n1, x_p1, y_p1, z_p1, r_p1;

  // GEM Station 2 = GE2/1 Short
  std::unique_ptr<TGraph>  GE2Sin_Plus_XY_All, GE2Sin_Minus_XY_All, GE2Sout_Plus_XY_All, GE2Sout_Minus_XY_All, GE2Sout_Plus_YZ_All, GE2Sout_Minus_YZ_All, GE2Sin_Plus_YZ_All, GE2Sin_Minus_YZ_All, GE2S_Plus_YZ_All, GE2S_Minus_YZ_All;
  std::unique_ptr<TCanvas> Canvas_GE2S_Plus_XY, Canvas_GE2S_Minus_XY, Canvas_GE2S_Plus_YZ, Canvas_GE2S_Minus_YZ;
  std::vector<double> x_n2si, y_n2si, z_n2si, r_n2si, x_n2so, y_n2so, z_n2so, r_n2so;
  std::vector<double> x_p2si, y_p2si, z_p2si, r_p2si, x_p2so, y_p2so, z_p2so, r_p2so;
  std::vector<double> x_n2s, y_n2s, z_n2s, r_n2s, x_p2s, y_p2s, z_p2s, r_p2s;

  // GEM Station 3 = GE2/1 Long
  std::unique_ptr<TGraph>  GE2Lin_Plus_XY_All, GE2Lin_Minus_XY_All, GE2Lout_Plus_XY_All, GE2Lout_Minus_XY_All, GE2Lout_Plus_YZ_All, GE2Lout_Minus_YZ_All, GE2Lin_Plus_YZ_All, GE2Lin_Minus_YZ_All, GE2L_Plus_YZ_All, GE2L_Minus_YZ_All;
  std::unique_ptr<TCanvas> Canvas_GE2L_Plus_XY, Canvas_GE2L_Minus_XY, Canvas_GE2L_Plus_YZ, Canvas_GE2L_Minus_YZ;
  std::vector<double> x_n2li, y_n2li, z_n2li, r_n2li, x_n2lo, y_n2lo, z_n2lo, r_n2lo;
  std::vector<double> x_p2li, y_p2li, z_p2li, r_p2li, x_p2lo, y_p2lo, z_p2lo, r_p2lo;
  std::vector<double> x_n2l, y_n2l, z_n2l, r_n2l, x_p2l, y_p2l, z_p2l, r_p2l;

  std::unique_ptr<TH1F> BX_RecHits_GE2Sin_Plus,  BX_RecHits_GE2Sout_Plus,  BX_RecHits_GE2Sin_Minus,  BX_RecHits_GE2Sout_Minus;
  std::unique_ptr<TH1F> ST_RecHits_GE2Sin_Plus, ST_RecHits_GE2Sout_Plus, ST_RecHits_GE2Sin_Minus, ST_RecHits_GE2Sout_Minus;
  std::unique_ptr<TH1F> CL_RecHits_GE2Sin_Plus, CL_RecHits_GE2Sout_Plus, CL_RecHits_GE2Sin_Minus, CL_RecHits_GE2Sout_Minus;

  std::unique_ptr<TH1F> BX_RecHits_GE2Lin_Plus,  BX_RecHits_GE2Lout_Plus,  BX_RecHits_GE2Lin_Minus,  BX_RecHits_GE2Lout_Minus;
  std::unique_ptr<TH1F> ST_RecHits_GE2Lin_Plus, ST_RecHits_GE2Lout_Plus, ST_RecHits_GE2Lin_Minus, ST_RecHits_GE2Lout_Minus;
  std::unique_ptr<TH1F> CL_RecHits_GE2Lin_Plus, CL_RecHits_GE2Lout_Plus, CL_RecHits_GE2Lin_Minus, CL_RecHits_GE2Lout_Minus;

};

//
// constants, enums and typedefs
//
int n_bx  = 11;   double n1_bx  = -5.5,  n2_bx  = 5.5;
int n_st  = 501;  double n1_st  = 0,     n2_st  = 500;
int n_cl  = 26;   double n1_cl  = -0.5,  n2_cl  = 25.5;

//
// static data member definitions
//

//
// constructors and destructor
//
TestGEMRecHitAnalyzer::TestGEMRecHitAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  GEMRecHit_Token = consumes<GEMRecHitCollection>(edm::InputTag("gemRecHits"));
  rootFileName  = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));
  
  // GE1/1
  BX_RecHits_GE1in_Plus    = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1in_Plus",   "BX_RecHits_GE1in_Plus",   n_bx, n1_bx, n2_bx));
  BX_RecHits_GE1out_Plus   = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1out_Plus",  "BX_RecHits_GE1out_Plus",  n_bx, n1_bx, n2_bx));
  BX_RecHits_GE1in_Minus   = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1in_Minus",  "BX_RecHits_GE1in_Minus",  n_bx, n1_bx, n2_bx));
  BX_RecHits_GE1out_Minus  = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1out_Minus", "BX_RecHits_GE1out_Minus", n_bx, n1_bx, n2_bx));

  ST_RecHits_GE1in_Plus   = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1in_Plus",   "ST_RecHits_GE1in_Plus",   n_st, n1_st, n2_st));
  ST_RecHits_GE1out_Plus  = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1out_Plus",  "ST_RecHits_GE1out_Plus",  n_st, n1_st, n2_st));   
  ST_RecHits_GE1in_Minus  = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1in_Minus",  "ST_RecHits_GE1in_Minus",  n_st, n1_st, n2_st));
  ST_RecHits_GE1out_Minus = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1out_Minus", "ST_RecHits_GE1out_Minus", n_st, n1_st, n2_st));

  CL_RecHits_GE1in_Plus   = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1in_Plus",   "CL_RecHits_GE1in_Plus",   n_cl, n1_cl, n2_cl));
  CL_RecHits_GE1out_Plus  = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1out_Plus",  "CL_RecHits_GE1out_Plus",  n_cl, n1_cl, n2_cl));   
  CL_RecHits_GE1in_Minus  = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1in_Minus",  "CL_RecHits_GE1in_Minus",  n_cl, n1_cl, n2_cl));
  CL_RecHits_GE1out_Minus = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1out_Minus", "CL_RecHits_GE1out_Minus", n_cl, n1_cl, n2_cl));

  // GE2/1 Short
  BX_RecHits_GE2Sin_Plus    = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Sin_Plus",   "BX_RecHits_GE2Sin_Plus",   n_bx, n1_bx, n2_bx));
  BX_RecHits_GE2Sout_Plus   = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Sout_Plus",  "BX_RecHits_GE2Sout_Plus",  n_bx, n1_bx, n2_bx));
  BX_RecHits_GE2Sin_Minus   = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Sin_Minus",  "BX_RecHits_GE2Sin_Minus",  n_bx, n1_bx, n2_bx));
  BX_RecHits_GE2Sout_Minus  = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Sout_Minus", "BX_RecHits_GE2Sout_Minus", n_bx, n1_bx, n2_bx));

  ST_RecHits_GE2Sin_Plus   = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Sin_Plus",   "ST_RecHits_GE2Sin_Plus",   n_st, n1_st, n2_st));
  ST_RecHits_GE2Sout_Plus  = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Sout_Plus",  "ST_RecHits_GE2Sout_Plus",  n_st, n1_st, n2_st));   
  ST_RecHits_GE2Sin_Minus  = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Sin_Minus",  "ST_RecHits_GE2Sin_Minus",  n_st, n1_st, n2_st));
  ST_RecHits_GE2Sout_Minus = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Sout_Minus", "ST_RecHits_GE2Sout_Minus", n_st, n1_st, n2_st));

  CL_RecHits_GE2Sin_Plus   = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Sin_Plus",   "CL_RecHits_GE2Sin_Plus",   n_cl, n1_cl, n2_cl));
  CL_RecHits_GE2Sout_Plus  = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Sout_Plus",  "CL_RecHits_GE2Sout_Plus",  n_cl, n1_cl, n2_cl));   
  CL_RecHits_GE2Sin_Minus  = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Sin_Minus",  "CL_RecHits_GE2Sin_Minus",  n_cl, n1_cl, n2_cl));
  CL_RecHits_GE2Sout_Minus = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Sout_Minus", "CL_RecHits_GE2Sout_Minus", n_cl, n1_cl, n2_cl));

  // GE2/1 Long
  BX_RecHits_GE2Lin_Plus    = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Lin_Plus",   "BX_RecHits_GE2Lin_Plus",   n_bx, n1_bx, n2_bx));
  BX_RecHits_GE2Lout_Plus   = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Lout_Plus",  "BX_RecHits_GE2Lout_Plus",  n_bx, n1_bx, n2_bx));
  BX_RecHits_GE2Lin_Minus   = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Lin_Minus",  "BX_RecHits_GE2Lin_Minus",  n_bx, n1_bx, n2_bx));
  BX_RecHits_GE2Lout_Minus  = std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE2Lout_Minus", "BX_RecHits_GE2Lout_Minus", n_bx, n1_bx, n2_bx));

  ST_RecHits_GE2Lin_Plus   = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Lin_Plus",   "ST_RecHits_GE2Lin_Plus",   n_st, n1_st, n2_st));
  ST_RecHits_GE2Lout_Plus  = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Lout_Plus",  "ST_RecHits_GE2Lout_Plus",  n_st, n1_st, n2_st));   
  ST_RecHits_GE2Lin_Minus  = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Lin_Minus",  "ST_RecHits_GE2Lin_Minus",  n_st, n1_st, n2_st));
  ST_RecHits_GE2Lout_Minus = std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE2Lout_Minus", "ST_RecHits_GE2Lout_Minus", n_st, n1_st, n2_st));

  CL_RecHits_GE2Lin_Plus   = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Lin_Plus",   "CL_RecHits_GE2Lin_Plus",   n_cl, n1_cl, n2_cl));
  CL_RecHits_GE2Lout_Plus  = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Lout_Plus",  "CL_RecHits_GE2Lout_Plus",  n_cl, n1_cl, n2_cl));   
  CL_RecHits_GE2Lin_Minus  = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Lin_Minus",  "CL_RecHits_GE2Lin_Minus",  n_cl, n1_cl, n2_cl));
  CL_RecHits_GE2Lout_Minus = std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE2Lout_Minus", "CL_RecHits_GE2Lout_Minus", n_cl, n1_cl, n2_cl));

}


TestGEMRecHitAnalyzer::~TestGEMRecHitAnalyzer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
 
  int mar1 = 4;
  int mar2 = 5;
  int col1 = 2;
  int col2 = 4;

  outputfile->cd();

  BX_RecHits_GE1in_Plus->Write();
  BX_RecHits_GE1out_Plus->Write();
  BX_RecHits_GE1in_Minus->Write();
  BX_RecHits_GE1out_Minus->Write();
  ST_RecHits_GE1in_Plus->Write();
  ST_RecHits_GE1out_Plus->Write();
  ST_RecHits_GE1in_Minus->Write();
  ST_RecHits_GE1out_Minus->Write();
  CL_RecHits_GE1in_Plus->Write();
  CL_RecHits_GE1out_Plus->Write();
  CL_RecHits_GE1in_Minus->Write();
  CL_RecHits_GE1out_Minus->Write();

  BX_RecHits_GE2Sin_Plus->Write();
  BX_RecHits_GE2Sout_Plus->Write();
  BX_RecHits_GE2Sin_Minus->Write();
  BX_RecHits_GE2Sout_Minus->Write();
  ST_RecHits_GE2Sin_Plus->Write();
  ST_RecHits_GE2Sout_Plus->Write();
  ST_RecHits_GE2Sin_Minus->Write();
  ST_RecHits_GE2Sout_Minus->Write();
  CL_RecHits_GE2Sin_Plus->Write();
  CL_RecHits_GE2Sout_Plus->Write();
  CL_RecHits_GE2Sin_Minus->Write();
  CL_RecHits_GE2Sout_Minus->Write();

  BX_RecHits_GE2Lin_Plus->Write();
  BX_RecHits_GE2Lout_Plus->Write();
  BX_RecHits_GE2Lin_Minus->Write();
  BX_RecHits_GE2Lout_Minus->Write();
  ST_RecHits_GE2Lin_Plus->Write();
  ST_RecHits_GE2Lout_Plus->Write();
  ST_RecHits_GE2Lin_Minus->Write();
  ST_RecHits_GE2Lout_Minus->Write();
  CL_RecHits_GE2Lin_Plus->Write();
  CL_RecHits_GE2Lout_Plus->Write();
  CL_RecHits_GE2Lin_Minus->Write();
  CL_RecHits_GE2Lout_Minus->Write();

  BX_RecHits_GE1 = std::unique_ptr<TCanvas>(new TCanvas("BX_RecHits_GE1", "BX_RecHits_GE1", 800, 600));
  ST_RecHits_GE1 = std::unique_ptr<TCanvas>(new TCanvas("ST_RecHits_GE1", "ST_RecHits_GE1", 800, 600));
  CL_RecHits_GE1 = std::unique_ptr<TCanvas>(new TCanvas("CL_RecHits_GE1", "CL_RecHits_GE1", 800, 600));

  BX_RecHits_GE1->cd();  BX_RecHits_GE1->Divide(2,2);
  BX_RecHits_GE1->cd(1); BX_RecHits_GE1in_Plus->Draw();  BX_RecHits_GE1in_Plus->GetXaxis()->SetTitle("BX [-]");  BX_RecHits_GE1in_Plus->GetYaxis()->SetTitle("entries [-]");   BX_RecHits_GE1in_Plus->SetTitle("GE+1in RecHits");
  BX_RecHits_GE1->cd(2); BX_RecHits_GE1out_Plus->Draw(); BX_RecHits_GE1out_Plus->GetXaxis()->SetTitle("BX [-]"); BX_RecHits_GE1out_Plus->GetYaxis()->SetTitle("entries [-]");  BX_RecHits_GE1out_Plus->SetTitle("GE+1out RecHits");
  BX_RecHits_GE1->cd(3); BX_RecHits_GE1in_Minus->Draw(); BX_RecHits_GE1in_Minus->GetXaxis()->SetTitle("BX [-]"); BX_RecHits_GE1in_Minus->GetYaxis()->SetTitle("entries [-]");  BX_RecHits_GE1in_Minus->SetTitle("GE-1in RecHits");
  BX_RecHits_GE1->cd(4); BX_RecHits_GE1out_Minus->Draw();BX_RecHits_GE1out_Minus->GetXaxis()->SetTitle("BX [-]");BX_RecHits_GE1out_Minus->GetYaxis()->SetTitle("entries [-]"); BX_RecHits_GE1out_Minus->SetTitle("GE-1out RecHits");

  ST_RecHits_GE1->cd();  ST_RecHits_GE1->Divide(2,2);
  ST_RecHits_GE1->cd(1); ST_RecHits_GE1in_Plus->Draw();   ST_RecHits_GE1in_Plus->GetXaxis()->SetTitle("First Strip of Cluster [-]");  ST_RecHits_GE1in_Plus->GetYaxis()->SetTitle("entries [-]");   ST_RecHits_GE1in_Plus->SetTitle("GE+1in RecHits");
  ST_RecHits_GE1->cd(2); ST_RecHits_GE1out_Plus->Draw();  ST_RecHits_GE1out_Plus->GetXaxis()->SetTitle("First Strip of Cluster [-]"); ST_RecHits_GE1out_Plus->GetYaxis()->SetTitle("entries [-]");  ST_RecHits_GE1out_Plus->SetTitle("GE+1out RecHits");
  ST_RecHits_GE1->cd(3); ST_RecHits_GE1in_Minus->Draw();  ST_RecHits_GE1in_Minus->GetXaxis()->SetTitle("First Strip of Cluster [-]"); ST_RecHits_GE1in_Minus->GetYaxis()->SetTitle("entries [-]");  ST_RecHits_GE1in_Minus->SetTitle("GE-1in RecHits");
  ST_RecHits_GE1->cd(4); ST_RecHits_GE1out_Minus->Draw(); ST_RecHits_GE1out_Minus->GetXaxis()->SetTitle("First Strip of Cluster [-]");ST_RecHits_GE1out_Minus->GetYaxis()->SetTitle("entries [-]"); ST_RecHits_GE1out_Minus->SetTitle("GE-1out RecHits");

  CL_RecHits_GE1->cd();  CL_RecHits_GE1->Divide(2,2);
  CL_RecHits_GE1->cd(1); CL_RecHits_GE1in_Plus->Draw();   CL_RecHits_GE1in_Plus->GetXaxis()->SetTitle("Clustersize [-]");  CL_RecHits_GE1in_Plus->GetYaxis()->SetTitle("entries [-]");   CL_RecHits_GE1in_Plus->SetTitle("GE+1in RecHits");
  CL_RecHits_GE1->cd(2); CL_RecHits_GE1out_Plus->Draw();  CL_RecHits_GE1out_Plus->GetXaxis()->SetTitle("Clustersize [-]"); CL_RecHits_GE1out_Plus->GetYaxis()->SetTitle("entries [-]");  CL_RecHits_GE1out_Plus->SetTitle("GE+1out RecHits");
  CL_RecHits_GE1->cd(3); CL_RecHits_GE1in_Minus->Draw();  CL_RecHits_GE1in_Minus->GetXaxis()->SetTitle("Clustersize [-]"); CL_RecHits_GE1in_Minus->GetYaxis()->SetTitle("entries [-]");  CL_RecHits_GE1in_Minus->SetTitle("GE-1in RecHits");
  CL_RecHits_GE1->cd(4); CL_RecHits_GE1out_Minus->Draw(); CL_RecHits_GE1out_Minus->GetXaxis()->SetTitle("Clustersize [-]");CL_RecHits_GE1out_Minus->GetYaxis()->SetTitle("entries [-]"); CL_RecHits_GE1out_Minus->SetTitle("GE-1out RecHits");

  // =======================
  // GE1/1 XY and RZ graphs
  // =======================
  const int n_n1i = x_n1i.size();  double x_an1i[n_n1i]; double y_an1i[n_n1i];  double z_an1i[n_n1i]; double r_an1i[n_n1i];
  const int n_n1o = x_n1o.size();  double x_an1o[n_n1o]; double y_an1o[n_n1o];  double z_an1o[n_n1o]; double r_an1o[n_n1o];
  const int n_p1i = x_p1i.size();  double x_ap1i[n_p1i]; double y_ap1i[n_p1i];  double z_ap1i[n_p1i]; double r_ap1i[n_p1i];
  const int n_p1o = x_p1o.size();  double x_ap1o[n_p1o]; double y_ap1o[n_p1o];  double z_ap1o[n_p1o]; double r_ap1o[n_p1o];
  const int n_n1  = x_n1.size();   /*double x_an1[n_n1]; double y_an1[n_n1];*/  double z_an1[n_n1]; double r_an1[n_n1];
  const int n_p1  = x_p1.size();   /*double x_ap1[n_p1]; double y_ap1[n_p1];*/  double z_ap1[n_p1]; double r_ap1[n_p1];

  for(int i=0; i< n_n1i; ++i) { x_an1i[i] = x_n1i[i]; y_an1i[i] = y_n1i[i]; z_an1i[i] = z_n1i[i]; r_an1i[i] = r_n1i[i];}
  for(int i=0; i< n_n1o; ++i) { x_an1o[i] = x_n1o[i]; y_an1o[i] = y_n1o[i]; z_an1o[i] = z_n1o[i]; r_an1o[i] = r_n1o[i];}
  for(int i=0; i< n_p1i; ++i) { x_ap1i[i] = x_p1i[i]; y_ap1i[i] = y_p1i[i]; z_ap1i[i] = z_p1i[i]; r_ap1i[i] = r_p1i[i];}
  for(int i=0; i< n_p1o; ++i) { x_ap1o[i] = x_p1o[i]; y_ap1o[i] = y_p1o[i]; z_ap1o[i] = z_p1o[i]; r_ap1o[i] = r_p1o[i];}
  for(int i=0; i< n_n1; ++i)  { /*x_an1[i] = x_n1[i]; y_an1[i] = y_n1[i];*/ z_an1[i] = z_n1[i]; r_an1[i] = r_n1[i];}
  for(int i=0; i< n_p1; ++i)  { /*x_ap1[i] = x_p1[i]; y_ap1[i] = y_p1[i];*/ z_ap1[i] = z_p1[i]; r_ap1[i] = r_p1[i];}

  GE1in_Minus_XY_All  = std::unique_ptr<TGraph>(new TGraph(n_n1i, x_an1i, y_an1i)); std::cout<<"GE-1in All SimHits: "<<n_n1i<<std::endl;
  GE1out_Minus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_n1o, x_an1o, y_an1o)); std::cout<<"GE-1out All SimHits: "<<n_n1o<<std::endl;
  GE1in_Plus_XY_All   = std::unique_ptr<TGraph>(new TGraph(n_p1i, x_ap1i, y_ap1i)); std::cout<<"GE+1in All SimHits: "<<n_p1i<<std::endl;
  GE1out_Plus_XY_All  = std::unique_ptr<TGraph>(new TGraph(n_p1o, x_ap1o, y_ap1o)); std::cout<<"GE+1out All SimHits: "<<n_p1o<<std::endl;

  GE1in_Minus_YZ_All  = std::unique_ptr<TGraph>(new TGraph(n_n1i, z_an1i, r_an1i));
  GE1out_Minus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_n1o, z_an1o, r_an1o));
  GE1in_Plus_YZ_All   = std::unique_ptr<TGraph>(new TGraph(n_p1i, z_ap1i, r_ap1i));
  GE1out_Plus_YZ_All  = std::unique_ptr<TGraph>(new TGraph(n_p1o, z_ap1o, r_ap1o));

  GE1_Minus_YZ_All    = std::unique_ptr<TGraph>(new TGraph(n_n1, z_an1, r_an1));
  GE1_Plus_YZ_All     = std::unique_ptr<TGraph>(new TGraph(n_p1, z_ap1, r_ap1));

  Canvas_GE1_Plus_XY   = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Plus_XY",   "Canvas_GE1_Plus_XY", 800, 600));
  Canvas_GE1_Minus_XY  = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Minus_XY",  "Canvas_GE1_Minus_XY", 800, 600));
  Canvas_GE1_Plus_YZ   = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Plus_YZ",   "Canvas_GE1_Plus_YZ", 600, 800));
  Canvas_GE1_Minus_YZ  = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Minus_YZ",  "Canvas_GE1_Minus_YZ", 600, 800));

  // XY and RZ Graphs
  Canvas_GE1_Plus_XY->cd();
  GE1in_Plus_XY_All->SetMarkerStyle(mar1);  GE1in_Plus_XY_All->SetMarkerColor(col1);  GE1in_Plus_XY_All->Draw("AP");  
  GE1out_Plus_XY_All->SetMarkerStyle(mar2); GE1out_Plus_XY_All->SetMarkerColor(col2); GE1out_Plus_XY_All->Draw("Psame");   
  GE1in_Plus_XY_All->GetXaxis()->SetTitle("X [cm]");  GE1in_Plus_XY_All->GetYaxis()->SetTitle("Y [cm]");   GE1in_Plus_XY_All->SetTitle("GE+1/1 RecHits"); 

  Canvas_GE1_Minus_XY->cd();
  GE1in_Minus_XY_All->SetMarkerStyle(mar1);  GE1in_Minus_XY_All->SetMarkerColor(col1);  GE1in_Minus_XY_All->Draw("AP"); 
  GE1out_Minus_XY_All->SetMarkerStyle(mar2); GE1out_Minus_XY_All->SetMarkerColor(col2); GE1out_Minus_XY_All->Draw("Psame"); 
  GE1in_Minus_XY_All->GetXaxis()->SetTitle("X [cm]"); GE1in_Minus_XY_All->GetYaxis()->SetTitle("Y [cm]");  GE1in_Minus_XY_All->SetTitle("GE-1/1 RecHits");

  Canvas_GE1_Plus_YZ->cd();  
  GE1_Plus_YZ_All->SetMarkerStyle(0);     GE1_Plus_YZ_All->SetMarkerColor(0);  GE1_Plus_YZ_All->Draw("AP"); GE1_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]");    GE1_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]");     GE1_Plus_YZ_All->SetTitle("GE+1/1 RecHits");
  GE1in_Plus_YZ_All->SetMarkerStyle(mar1);     GE1in_Plus_YZ_All->SetMarkerColor(col1);  GE1in_Plus_YZ_All->Draw("Psame");
  GE1out_Plus_YZ_All->SetMarkerStyle(mar2);    GE1out_Plus_YZ_All->SetMarkerColor(col2); GE1out_Plus_YZ_All->Draw("Psame");
  // GE1in_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]");    GE1in_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]");     GE1in_Plus_YZ_All->SetTitle("GE1 Plus RecHits");
  // GE1in_Plus_YZ_All->GetXaxis()->SetRangeUser(563,573);
  // TLatex latex1; latex1.SetNDC();  latex1.SetTextAlign(23); 
  // latex1.SetTextSize(0.03); latex1.DrawLatex(0.175,0.50,"#color[15]{RE+1/2}"); latex1.DrawLatex(0.325,0.50,"#color[15]{RE+2/2}"); latex1.DrawLatex(0.675,0.50,"#color[15]{RE+3/2}"); latex1.DrawLatex(0.825,0.50,"#color[15]{RE+4/2}");
  // latex1.SetTextSize(0.03); latex1.DrawLatex(0.175,0.875,"#color[15]{RE+1/3}"); latex1.DrawLatex(0.325,0.875,"#color[15]{RE+2/3}"); latex1.DrawLatex(0.675,0.875,"#color[15]{RE+3/3}"); latex1.DrawLatex(0.825,0.875,"#color[15]{RE+4/3}");

  Canvas_GE1_Minus_YZ->cd(); 
  GE1_Minus_YZ_All->SetMarkerStyle(0);   GE1_Minus_YZ_All->SetMarkerColor(0);  GE1_Minus_YZ_All->Draw("AP");   GE1_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]");   GE1_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]");    GE1_Minus_YZ_All->SetTitle("GE-1/1 RecHits"); 
  GE1in_Minus_YZ_All->SetMarkerStyle(mar1);   GE1in_Minus_YZ_All->SetMarkerColor(col1);  GE1in_Minus_YZ_All->Draw("Psame");  
  GE1out_Minus_YZ_All->SetMarkerStyle(mar2);  GE1out_Minus_YZ_All->SetMarkerColor(col2); GE1out_Minus_YZ_All->Draw("Psame");   
  // GE1in_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]");   GE1in_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]");    GE1in_Minus_YZ_All->SetTitle("GE Minus RecHits");
  // GE1in_Minus_YZ_All->GetXaxis()->SetRangeUser(-573, -563);
  // TLatex latex2; latex2.SetNDC();  latex2.SetTextAlign(23); 
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.175,0.50,"#color[15]{RE-4/2}"); latex2.DrawLatex(0.325,0.50,"#color[15]{RE-3/2}"); latex2.DrawLatex(0.675,0.50,"#color[15]{RE-2/2}"); latex2.DrawLatex(0.825,0.50,"#color[15]{RE-1/2}");
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.175,0.875,"#color[15]{RE-4/3}"); latex2.DrawLatex(0.325,0.875,"#color[15]{RE-3/3}"); latex2.DrawLatex(0.675,0.875,"#color[15]{RE-2/3}"); latex2.DrawLatex(0.825,0.875,"#color[15]{RE-1/3}");


  // ============================
  // GE2/1 Short XY and RZ graphs
  // ============================
  const int n_n2si = x_n2si.size();  double x_an2si[n_n2si]; double y_an2si[n_n2si];  double z_an2si[n_n2si]; double r_an2si[n_n2si];
  const int n_n2so = x_n2so.size();  double x_an2so[n_n2so]; double y_an2so[n_n2so];  double z_an2so[n_n2so]; double r_an2so[n_n2so];
  const int n_p2si = x_p2si.size();  double x_ap2si[n_p2si]; double y_ap2si[n_p2si];  double z_ap2si[n_p2si]; double r_ap2si[n_p2si];
  const int n_p2so = x_p2so.size();  double x_ap2so[n_p2so]; double y_ap2so[n_p2so];  double z_ap2so[n_p2so]; double r_ap2so[n_p2so];
  const int n_n2s  = x_n2s.size();   /*double x_an2s[n_n2s]; double y_an2s[n_n2s];*/  double z_an2s[n_n2s]; double r_an2s[n_n2s];
  const int n_p2s  = x_p2s.size();   /*double x_ap2s[n_p2s]; double y_ap2s[n_p2s];*/  double z_ap2s[n_p2s]; double r_ap2s[n_p2s];

  for(int i=0; i< n_n2si; ++i) { x_an2si[i] = x_n2si[i]; y_an2si[i] = y_n2si[i]; z_an2si[i] = z_n2si[i]; r_an2si[i] = r_n2si[i];}
  for(int i=0; i< n_n2so; ++i) { x_an2so[i] = x_n2so[i]; y_an2so[i] = y_n2so[i]; z_an2so[i] = z_n2so[i]; r_an2so[i] = r_n2so[i];}
  for(int i=0; i< n_p2si; ++i) { x_ap2si[i] = x_p2si[i]; y_ap2si[i] = y_p2si[i]; z_ap2si[i] = z_p2si[i]; r_ap2si[i] = r_p2si[i];}
  for(int i=0; i< n_p2so; ++i) { x_ap2so[i] = x_p2so[i]; y_ap2so[i] = y_p2so[i]; z_ap2so[i] = z_p2so[i]; r_ap2so[i] = r_p2so[i];}
  for(int i=0; i< n_n2s; ++i)  { /*x_an2s[i] = x_n2s[i]; y_an2s[i] = y_n2s[i];*/ z_an2s[i] = z_n2s[i]; r_an2s[i] = r_n2s[i];}
  for(int i=0; i< n_p2s; ++i)  { /*x_ap2s[i] = x_p2s[i]; y_ap2s[i] = y_p2s[i];*/ z_ap2s[i] = z_p2s[i]; r_ap2s[i] = r_p2s[i];}

  GE2Sin_Minus_XY_All  = std::unique_ptr<TGraph>(new TGraph(n_n2si, x_an2si, y_an2si)); std::cout<<"GE-2sin All SimHits: "<<n_n2si<<std::endl;
  GE2Sout_Minus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_n2so, x_an2so, y_an2so)); std::cout<<"GE-2sout All SimHits: "<<n_n2so<<std::endl;
  GE2Sin_Plus_XY_All   = std::unique_ptr<TGraph>(new TGraph(n_p2si, x_ap2si, y_ap2si)); std::cout<<"GE+2sin All SimHits: "<<n_p2si<<std::endl;
  GE2Sout_Plus_XY_All  = std::unique_ptr<TGraph>(new TGraph(n_p2so, x_ap2so, y_ap2so)); std::cout<<"GE+2sout All SimHits: "<<n_p2so<<std::endl;

  GE2Sin_Minus_YZ_All  = std::unique_ptr<TGraph>(new TGraph(n_n2si, z_an2si, r_an2si));
  GE2Sout_Minus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_n2so, z_an2so, r_an2so));
  GE2Sin_Plus_YZ_All   = std::unique_ptr<TGraph>(new TGraph(n_p2si, z_ap2si, r_ap2si));
  GE2Sout_Plus_YZ_All  = std::unique_ptr<TGraph>(new TGraph(n_p2so, z_ap2so, r_ap2so));

  GE2S_Minus_YZ_All    = std::unique_ptr<TGraph>(new TGraph(n_n2s, z_an2s, r_an2s));
  GE2S_Plus_YZ_All     = std::unique_ptr<TGraph>(new TGraph(n_p2s, z_ap2s, r_ap2s));

  Canvas_GE2S_Plus_XY   = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2S_Plus_XY",   "Canvas_GE2S_Plus_XY", 800, 600));
  Canvas_GE2S_Minus_XY  = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2S_Minus_XY",  "Canvas_GE2S_Minus_XY", 800, 600));
  Canvas_GE2S_Plus_YZ   = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2S_Plus_YZ",   "Canvas_GE2S_Plus_YZ", 600, 800));
  Canvas_GE2S_Minus_YZ  = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2S_Minus_YZ",  "Canvas_GE2S_Minus_YZ", 600, 800));

  // XY and RZ Graphs
  Canvas_GE2S_Plus_XY->cd();
  GE2Sin_Plus_XY_All->SetMarkerStyle(mar1);  GE2Sin_Plus_XY_All->SetMarkerColor(col1);  GE2Sin_Plus_XY_All->Draw("AP");  
  GE2Sout_Plus_XY_All->SetMarkerStyle(mar2); GE2Sout_Plus_XY_All->SetMarkerColor(col2); GE2Sout_Plus_XY_All->Draw("Psame");   
  GE2Sin_Plus_XY_All->GetXaxis()->SetTitle("X [cm]");  GE2Sin_Plus_XY_All->GetYaxis()->SetTitle("Y [cm]");   GE2Sin_Plus_XY_All->SetTitle("GE+2/1Short RecHits"); 

  Canvas_GE2S_Minus_XY->cd();
  GE2Sin_Minus_XY_All->SetMarkerStyle(mar1);  GE2Sin_Minus_XY_All->SetMarkerColor(col1);  GE2Sin_Minus_XY_All->Draw("AP"); 
  GE2Sout_Minus_XY_All->SetMarkerStyle(mar2); GE2Sout_Minus_XY_All->SetMarkerColor(col2); GE2Sout_Minus_XY_All->Draw("Psame"); 
  GE2Sin_Minus_XY_All->GetXaxis()->SetTitle("X [cm]"); GE2Sin_Minus_XY_All->GetYaxis()->SetTitle("Y [cm]");  GE2Sin_Minus_XY_All->SetTitle("GE-2/1Short RecHits");

  Canvas_GE2S_Plus_YZ->cd();  
  GE2S_Plus_YZ_All->SetMarkerStyle(0); GE2S_Plus_YZ_All->SetMarkerColor(0); GE2S_Plus_YZ_All->Draw("AP"); GE2S_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]"); GE2S_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]"); GE2S_Plus_YZ_All->SetTitle("GE+2/1Short RecHits");
  GE2Sin_Plus_YZ_All->SetMarkerStyle(mar1);    GE2Sin_Plus_YZ_All->SetMarkerColor(col1);  GE2Sin_Plus_YZ_All->Draw("Psame");
  GE2Sout_Plus_YZ_All->SetMarkerStyle(mar2);    GE2Sout_Plus_YZ_All->SetMarkerColor(col2); GE2Sout_Plus_YZ_All->Draw("Psame");
  // GE2Sin_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]");    GE2Sin_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]");     GE2Sin_Plus_YZ_All->SetTitle("GE2S Plus RecHits");
  // GE2Sin_Plus_YZ_All->GetXaxis()->SetRangeUser(563,573);
  // TLatex latex2s; latex2s.SetNDC();  latex2s.SetTextAlign(23); 
  // latex2s.SetTextSize(0.03); latex2s.DrawLatex(0.2s75,0.50,"#color[2s5]{RE+2s/2}");  latex2s.DrawLatex(0.325,0.50,"#color[2s5]{RE+2/2}");  latex2s.DrawLatex(0.675,0.50,"#color[2s5]{RE+3/2}");  latex2s.DrawLatex(0.825,0.50,"#color[2s5]{RE+4/2}");
  // latex2s.SetTextSize(0.03); latex2s.DrawLatex(0.2s75,0.875,"#color[2s5]{RE+2s/3}"); latex2s.DrawLatex(0.325,0.875,"#color[2s5]{RE+2/3}"); latex2s.DrawLatex(0.675,0.875,"#color[2s5]{RE+3/3}"); latex2s.DrawLatex(0.825,0.875,"#color[2s5]{RE+4/3}");

  Canvas_GE2S_Minus_YZ->cd(); 
  GE2S_Minus_YZ_All->SetMarkerStyle(0); GE2S_Minus_YZ_All->SetMarkerColor(0); GE2S_Minus_YZ_All->Draw("AP"); GE2S_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]"); GE2S_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]"); GE2S_Minus_YZ_All->SetTitle("GE-2/1Short RecHits"); 
  GE2Sin_Minus_YZ_All->SetMarkerStyle(mar1);  GE2Sin_Minus_YZ_All->SetMarkerColor(col1);  GE2Sin_Minus_YZ_All->Draw("Psame");  
  GE2Sout_Minus_YZ_All->SetMarkerStyle(mar2);  GE2Sout_Minus_YZ_All->SetMarkerColor(col2); GE2Sout_Minus_YZ_All->Draw("Psame");   
  // GE2Sin_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]");   GE2Sin_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]");    GE2Sin_Minus_YZ_All->SetTitle("GE Minus RecHits");
  // GE2Sin_Minus_YZ_All->GetXaxis()->SetRangeUser(-573, -563);
  // TLatex latex2; latex2.SetNDC();  latex2.SetTextAlign(23); 
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.2s75,0.50,"#color[2s5]{RE-4/2}"); latex2.DrawLatex(0.325,0.50,"#color[2s5]{RE-3/2}"); latex2.DrawLatex(0.675,0.50,"#color[2s5]{RE-2/2}"); latex2.DrawLatex(0.825,0.50,"#color[2s5]{RE-2s/2}");
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.2s75,0.875,"#color[2s5]{RE-4/3}"); latex2.DrawLatex(0.325,0.875,"#color[2s5]{RE-3/3}"); latex2.DrawLatex(0.675,0.875,"#color[2s5]{RE-2/3}"); latex2.DrawLatex(0.825,0.875,"#color[2s5]{RE-2s/3}");


  // ===========================
  // GE2/1 Long XY and RZ graphs
  // ===========================
  const int n_n2li = x_n2li.size();  double x_an2li[n_n2li]; double y_an2li[n_n2li];  double z_an2li[n_n2li]; double r_an2li[n_n2li];
  const int n_n2lo = x_n2lo.size();  double x_an2lo[n_n2lo]; double y_an2lo[n_n2lo];  double z_an2lo[n_n2lo]; double r_an2lo[n_n2lo];
  const int n_p2li = x_p2li.size();  double x_ap2li[n_p2li]; double y_ap2li[n_p2li];  double z_ap2li[n_p2li]; double r_ap2li[n_p2li];
  const int n_p2lo = x_p2lo.size();  double x_ap2lo[n_p2lo]; double y_ap2lo[n_p2lo];  double z_ap2lo[n_p2lo]; double r_ap2lo[n_p2lo];
  const int n_n2l  = x_n2l.size();   /*double x_an2l[n_n2l]; double y_an2l[n_n2l];*/  double z_an2l[n_n2l]; double r_an2l[n_n2l];
  const int n_p2l  = x_p2l.size();   /*double x_ap2l[n_p2l]; double y_ap2l[n_p2l];*/  double z_ap2l[n_p2l]; double r_ap2l[n_p2l];

  for(int i=0; i< n_n2li; ++i) { x_an2li[i] = x_n2li[i]; y_an2li[i] = y_n2li[i]; z_an2li[i] = z_n2li[i]; r_an2li[i] = r_n2li[i];}
  for(int i=0; i< n_n2lo; ++i) { x_an2lo[i] = x_n2lo[i]; y_an2lo[i] = y_n2lo[i]; z_an2lo[i] = z_n2lo[i]; r_an2lo[i] = r_n2lo[i];}
  for(int i=0; i< n_p2li; ++i) { x_ap2li[i] = x_p2li[i]; y_ap2li[i] = y_p2li[i]; z_ap2li[i] = z_p2li[i]; r_ap2li[i] = r_p2li[i];}
  for(int i=0; i< n_p2lo; ++i) { x_ap2lo[i] = x_p2lo[i]; y_ap2lo[i] = y_p2lo[i]; z_ap2lo[i] = z_p2lo[i]; r_ap2lo[i] = r_p2lo[i];}
  for(int i=0; i< n_n2l; ++i)  { /*x_an2l[i] = x_n2l[i]; y_an2l[i] = y_n2l[i];*/ z_an2l[i] = z_n2l[i]; r_an2l[i] = r_n2l[i];}
  for(int i=0; i< n_p2l; ++i)  { /*x_ap2l[i] = x_p2l[i]; y_ap2l[i] = y_p2l[i];*/ z_ap2l[i] = z_p2l[i]; r_ap2l[i] = r_p2l[i];}

  GE2Lin_Minus_XY_All  = std::unique_ptr<TGraph>(new TGraph(n_n2li, x_an2li, y_an2li)); std::cout<<"GE-2lin All SimHits: "<<n_n2li<<std::endl;
  GE2Lout_Minus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_n2lo, x_an2lo, y_an2lo)); std::cout<<"GE-2lout All SimHits: "<<n_n2lo<<std::endl;
  GE2Lin_Plus_XY_All   = std::unique_ptr<TGraph>(new TGraph(n_p2li, x_ap2li, y_ap2li)); std::cout<<"GE+2lin All SimHits: "<<n_p2li<<std::endl;
  GE2Lout_Plus_XY_All  = std::unique_ptr<TGraph>(new TGraph(n_p2lo, x_ap2lo, y_ap2lo)); std::cout<<"GE+2lout All SimHits: "<<n_p2lo<<std::endl;

  GE2Lin_Minus_YZ_All  = std::unique_ptr<TGraph>(new TGraph(n_n2li, z_an2li, r_an2li));
  GE2Lout_Minus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_n2lo, z_an2lo, r_an2lo));
  GE2Lin_Plus_YZ_All   = std::unique_ptr<TGraph>(new TGraph(n_p2li, z_ap2li, r_ap2li));
  GE2Lout_Plus_YZ_All  = std::unique_ptr<TGraph>(new TGraph(n_p2lo, z_ap2lo, r_ap2lo));

  GE2L_Minus_YZ_All    = std::unique_ptr<TGraph>(new TGraph(n_n2l, z_an2l, r_an2l));
  GE2L_Plus_YZ_All     = std::unique_ptr<TGraph>(new TGraph(n_p2l, z_ap2l, r_ap2l));

  Canvas_GE2L_Plus_XY   = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2L_Plus_XY",   "Canvas_GE2L_Plus_XY", 800, 600));
  Canvas_GE2L_Minus_XY  = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2L_Minus_XY",  "Canvas_GE2L_Minus_XY", 800, 600));
  Canvas_GE2L_Plus_YZ   = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2L_Plus_YZ",   "Canvas_GE2L_Plus_YZ", 600, 800));
  Canvas_GE2L_Minus_YZ  = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE2L_Minus_YZ",  "Canvas_GE2L_Minus_YZ", 600, 800));

  // XY and RZ Graphs
  Canvas_GE2L_Plus_XY->cd();
  GE2Lin_Plus_XY_All->SetMarkerStyle(mar1);  GE2Lin_Plus_XY_All->SetMarkerColor(col1);  GE2Lin_Plus_XY_All->Draw("AP");  
  GE2Lout_Plus_XY_All->SetMarkerStyle(mar2); GE2Lout_Plus_XY_All->SetMarkerColor(col2); GE2Lout_Plus_XY_All->Draw("Psame");   
  GE2Lin_Plus_XY_All->GetXaxis()->SetTitle("X [cm]");  GE2Lin_Plus_XY_All->GetYaxis()->SetTitle("Y [cm]");   GE2Lin_Plus_XY_All->SetTitle("GE+2/1Long RecHits"); 

  Canvas_GE2L_Minus_XY->cd();
  GE2Lin_Minus_XY_All->SetMarkerStyle(mar1);  GE2Lin_Minus_XY_All->SetMarkerColor(col1);  GE2Lin_Minus_XY_All->Draw("AP"); 
  GE2Lout_Minus_XY_All->SetMarkerStyle(mar2); GE2Lout_Minus_XY_All->SetMarkerColor(col2); GE2Lout_Minus_XY_All->Draw("Psame"); 
  GE2Lin_Minus_XY_All->GetXaxis()->SetTitle("X [cm]"); GE2Lin_Minus_XY_All->GetYaxis()->SetTitle("Y [cm]");  GE2Lin_Minus_XY_All->SetTitle("GE-2/1Long RecHits");

  Canvas_GE2L_Plus_YZ->cd();  
  GE2L_Plus_YZ_All->SetMarkerStyle(0); GE2L_Plus_YZ_All->SetMarkerColor(0); GE2L_Plus_YZ_All->Draw("AP"); GE2L_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]"); GE2L_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]"); GE2L_Plus_YZ_All->SetTitle("GE+2/1Long RecHits");
  GE2Lin_Plus_YZ_All->SetMarkerStyle(mar1);    GE2Lin_Plus_YZ_All->SetMarkerColor(col1);  GE2Lin_Plus_YZ_All->Draw("Psame");
  GE2Lout_Plus_YZ_All->SetMarkerStyle(mar2);    GE2Lout_Plus_YZ_All->SetMarkerColor(col2); GE2Lout_Plus_YZ_All->Draw("Psame");
  // GE2Lin_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]");    GE2Lin_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]");     GE2Lin_Plus_YZ_All->SetTitle("GE2L Plus RecHits");
  // GE2Lin_Plus_YZ_All->GetXaxis()->SetRangeUser(563,573);
  // TLatex latex2l; latex2l.SetNDC();  latex2l.SetTextAlign(23); 
  // latex2l.SetTextSize(0.03); latex2l.DrawLatex(0.2l75,0.50,"#color[2l5]{RE+2l/2}");  latex2l.DrawLatex(0.325,0.50,"#color[2l5]{RE+2/2}");  latex2l.DrawLatex(0.675,0.50,"#color[2l5]{RE+3/2}");  latex2l.DrawLatex(0.825,0.50,"#color[2l5]{RE+4/2}");
  // latex2l.SetTextSize(0.03); latex2l.DrawLatex(0.2l75,0.875,"#color[2l5]{RE+2l/3}"); latex2l.DrawLatex(0.325,0.875,"#color[2l5]{RE+2/3}"); latex2l.DrawLatex(0.675,0.875,"#color[2l5]{RE+3/3}"); latex2l.DrawLatex(0.825,0.875,"#color[2l5]{RE+4/3}");

  Canvas_GE2L_Minus_YZ->cd(); 
  GE2L_Minus_YZ_All->SetMarkerStyle(0); GE2L_Minus_YZ_All->SetMarkerColor(0); GE2L_Minus_YZ_All->Draw("AP"); GE2L_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]"); GE2L_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]"); GE2L_Minus_YZ_All->SetTitle("GE-2/1Long RecHits"); 
  GE2Lin_Minus_YZ_All->SetMarkerStyle(mar1);  GE2Lin_Minus_YZ_All->SetMarkerColor(col1);  GE2Lin_Minus_YZ_All->Draw("Psame");  
  GE2Lout_Minus_YZ_All->SetMarkerStyle(mar2);  GE2Lout_Minus_YZ_All->SetMarkerColor(col2); GE2Lout_Minus_YZ_All->Draw("Psame");   
  // GE2Lin_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]");   GE2Lin_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]");    GE2Lin_Minus_YZ_All->SetTitle("GE Minus RecHits");
  // GE2Lin_Minus_YZ_All->GetXaxis()->SetRangeUser(-573, -563);
  // TLatex latex2; latex2.SetNDC();  latex2.SetTextAlign(23); 
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.2l75,0.50,"#color[2l5]{RE-4/2}"); latex2.DrawLatex(0.325,0.50,"#color[2l5]{RE-3/2}"); latex2.DrawLatex(0.675,0.50,"#color[2l5]{RE-2/2}"); latex2.DrawLatex(0.825,0.50,"#color[2l5]{RE-2l/2}");
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.2l75,0.875,"#color[2l5]{RE-4/3}"); latex2.DrawLatex(0.325,0.875,"#color[2l5]{RE-3/3}"); latex2.DrawLatex(0.675,0.875,"#color[2l5]{RE-2/3}"); latex2.DrawLatex(0.825,0.875,"#color[2l5]{RE-2l/3}");



  // Save them all
  BX_RecHits_GE1->Write();
  ST_RecHits_GE1->Write();
  CL_RecHits_GE1->Write();
  Canvas_GE1_Plus_XY->Write();
  Canvas_GE1_Minus_XY->Write();
  Canvas_GE1_Plus_YZ->Write();
  Canvas_GE1_Minus_YZ->Write();
  Canvas_GE2S_Plus_XY->Write();
  Canvas_GE2S_Minus_XY->Write();
  Canvas_GE2S_Plus_YZ->Write();
  Canvas_GE2S_Minus_YZ->Write();
  Canvas_GE2L_Plus_XY->Write();
  Canvas_GE2L_Minus_XY->Write();
  Canvas_GE2L_Plus_YZ->Write();
  Canvas_GE2L_Minus_YZ->Write();
}


//
// member functions
//

// ------------ method called for each event  ------------
void
TestGEMRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  iSetup.get<MuonGeometryRecord>().get(gemGeom);

  // ================
  // GEM recHits
  // ================
  edm::Handle<GEMRecHitCollection> gemRecHits; 
  iEvent.getByToken(GEMRecHit_Token,gemRecHits);

  // count the number of GEM rechits
  int nGEM = 0;
  GEMRecHitCollection::const_iterator recHit;
  for (recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    nGEM++;
  }
   
  // std::cout<<"The Number of RecHits is "<<nGEM<<std::endl;       
  for (recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    GEMDetId rollId = (GEMDetId)(*recHit).gemId();
    LocalPoint recHitPos=recHit->localPosition();
    const GEMEtaPartition* rollasociated = gemGeom->etaPartition(rollId);
    const BoundPlane & GEMSurface = rollasociated->surface(); 
    GlobalPoint GEMGlobalPoint = GEMSurface.toGlobal(recHitPos);

    int region  = rollId.region();
    int station = rollId.station();
    // int ring    = rollId.ring();
    int layer   = rollId.layer();
    // int etapart = rollId.roll();
    // int chamber = rollId.chamber();

    // std::cout<<"GEM Rec Hit in [DetId] = ["<<rollId<<"] with BX = "<<recHit->BunchX()<<" and Global Position = "<<GEMGlobalPoint<<std::endl;

    int bx = recHit->BunchX();
    int cl = recHit->clusterSize();
    int st = recHit->firstClusterStrip();

    // GE+1/1 Positive Endcap
    if(region == 1 && station == 1) {
      x_p1.push_back(GEMGlobalPoint.x()); y_p1.push_back(GEMGlobalPoint.y()); z_p1.push_back(GEMGlobalPoint.z()); r_p1.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));
      // Layers
      if(layer==1) { BX_RecHits_GE1in_Plus->Fill(bx);  ST_RecHits_GE1in_Plus->Fill(st); CL_RecHits_GE1in_Plus->Fill(cl); x_p1i.push_back(GEMGlobalPoint.x()); y_p1i.push_back(GEMGlobalPoint.y()); z_p1i.push_back(GEMGlobalPoint.z()); 
	r_p1i.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
      if(layer==2) { BX_RecHits_GE1out_Plus->Fill(bx); ST_RecHits_GE1out_Plus->Fill(st); CL_RecHits_GE1out_Plus->Fill(cl); x_p1o.push_back(GEMGlobalPoint.x()); y_p1o.push_back(GEMGlobalPoint.y()); z_p1o.push_back(GEMGlobalPoint.z());
	r_p1o.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
    }
    // GE-1/1 Negative Endcap
    if(region == -1 && station == 1) {
      x_n1.push_back(GEMGlobalPoint.x()); y_n1.push_back(GEMGlobalPoint.y()); z_n1.push_back(GEMGlobalPoint.z()); r_n1.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));
      // Layers
      if(layer==1) { BX_RecHits_GE1in_Minus->Fill(bx);  ST_RecHits_GE1in_Minus->Fill(st);  CL_RecHits_GE1in_Minus->Fill(cl); x_n1i.push_back(GEMGlobalPoint.x()); y_n1i.push_back(GEMGlobalPoint.y()); z_n1i.push_back(GEMGlobalPoint.z());
	r_n1i.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
      if(layer==2) { BX_RecHits_GE1out_Minus->Fill(bx); ST_RecHits_GE1out_Minus->Fill(st); CL_RecHits_GE1out_Minus->Fill(cl); x_n1o.push_back(GEMGlobalPoint.x()); y_n1o.push_back(GEMGlobalPoint.y()); z_n1o.push_back(GEMGlobalPoint.z());
	r_n1o.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
    }

    // GE+2/1 Short Positive Endcap
    if(region == 1 && station == 2) {
      x_p2s.push_back(GEMGlobalPoint.x()); y_p2s.push_back(GEMGlobalPoint.y()); z_p2s.push_back(GEMGlobalPoint.z()); r_p2s.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));
      // Layers
      if(layer==1) { BX_RecHits_GE2Sin_Plus->Fill(bx);  ST_RecHits_GE2Sin_Plus->Fill(st); CL_RecHits_GE2Sin_Plus->Fill(cl); x_p2si.push_back(GEMGlobalPoint.x()); y_p2si.push_back(GEMGlobalPoint.y()); z_p2si.push_back(GEMGlobalPoint.z()); 
	r_p2si.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
      if(layer==2) { BX_RecHits_GE2Sout_Plus->Fill(bx); ST_RecHits_GE2Sout_Plus->Fill(st); CL_RecHits_GE2Sout_Plus->Fill(cl); x_p2so.push_back(GEMGlobalPoint.x()); y_p2so.push_back(GEMGlobalPoint.y()); z_p2so.push_back(GEMGlobalPoint.z());
	r_p2so.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
    }
    // GE-2/2s Short Negative Endcap
    if(region == -1 && station == 2) {
      x_n2s.push_back(GEMGlobalPoint.x()); y_n2s.push_back(GEMGlobalPoint.y()); z_n2s.push_back(GEMGlobalPoint.z()); r_n2s.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));
      // Layers
      if(layer==1) { BX_RecHits_GE2Sin_Minus->Fill(bx);  ST_RecHits_GE2Sin_Minus->Fill(st);  CL_RecHits_GE2Sin_Minus->Fill(cl); x_n2si.push_back(GEMGlobalPoint.x()); y_n2si.push_back(GEMGlobalPoint.y()); z_n2si.push_back(GEMGlobalPoint.z());
	r_n2si.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
      if(layer==2) { BX_RecHits_GE2Sout_Minus->Fill(bx); ST_RecHits_GE2Sout_Minus->Fill(st); CL_RecHits_GE2Sout_Minus->Fill(cl); x_n2so.push_back(GEMGlobalPoint.x()); y_n2so.push_back(GEMGlobalPoint.y()); z_n2so.push_back(GEMGlobalPoint.z());
	r_n2so.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
    }

    // GE+2/1 Long Positive Endcap
    if(region == 1 && station == 3) {
      x_p2l.push_back(GEMGlobalPoint.x()); y_p2l.push_back(GEMGlobalPoint.y()); z_p2l.push_back(GEMGlobalPoint.z()); r_p2l.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));
      // Layers
      if(layer==1) { BX_RecHits_GE2Lin_Plus->Fill(bx);  ST_RecHits_GE2Lin_Plus->Fill(st); CL_RecHits_GE2Lin_Plus->Fill(cl); x_p2li.push_back(GEMGlobalPoint.x()); y_p2li.push_back(GEMGlobalPoint.y()); z_p2li.push_back(GEMGlobalPoint.z()); 
	r_p2li.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
      if(layer==2) { BX_RecHits_GE2Lout_Plus->Fill(bx); ST_RecHits_GE2Lout_Plus->Fill(st); CL_RecHits_GE2Lout_Plus->Fill(cl); x_p2lo.push_back(GEMGlobalPoint.x()); y_p2lo.push_back(GEMGlobalPoint.y()); z_p2lo.push_back(GEMGlobalPoint.z());
	r_p2lo.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
    }
    // GE-2/2l Long Negative Endcap
    if(region == -1 && station == 3) {
      x_n2l.push_back(GEMGlobalPoint.x()); y_n2l.push_back(GEMGlobalPoint.y()); z_n2l.push_back(GEMGlobalPoint.z()); r_n2l.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));
      // Layers
      if(layer==1) { BX_RecHits_GE2Lin_Minus->Fill(bx);  ST_RecHits_GE2Lin_Minus->Fill(st);  CL_RecHits_GE2Lin_Minus->Fill(cl); x_n2li.push_back(GEMGlobalPoint.x()); y_n2li.push_back(GEMGlobalPoint.y()); z_n2li.push_back(GEMGlobalPoint.z());
	r_n2li.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
      if(layer==2) { BX_RecHits_GE2Lout_Minus->Fill(bx); ST_RecHits_GE2Lout_Minus->Fill(st); CL_RecHits_GE2Lout_Minus->Fill(cl); x_n2lo.push_back(GEMGlobalPoint.x()); y_n2lo.push_back(GEMGlobalPoint.y()); z_n2lo.push_back(GEMGlobalPoint.z());
	r_n2lo.push_back(sqrt(pow(GEMGlobalPoint.x(),2) + pow(GEMGlobalPoint.y(),2)));}
    }
  }

  // Print Separately Region +1 and -1
  for (recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    GEMDetId rollId = (GEMDetId)(*recHit).gemId();
    if(rollId.region()==-1) {
      LocalPoint recHitPos=recHit->localPosition();
      const GEMEtaPartition* rollasociated = gemGeom->etaPartition(rollId);
      const BoundPlane & GEMSurface = rollasociated->surface(); 
      GlobalPoint GEMGlobalPoint = GEMSurface.toGlobal(recHitPos);
      std::cout<<"GEM Rec Hit in [DetId] = ["<<rollId<<"] with BX = "<<recHit->BunchX()<<" and Global Position = "<<GEMGlobalPoint<<std::endl;
    }
  }
  for (recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    GEMDetId rollId = (GEMDetId)(*recHit).gemId();
    if(rollId.region()==+1) {
      LocalPoint recHitPos=recHit->localPosition();
      const GEMEtaPartition* rollasociated = gemGeom->etaPartition(rollId);
      const BoundPlane & GEMSurface = rollasociated->surface(); 
      GlobalPoint GEMGlobalPoint = GEMSurface.toGlobal(recHitPos);
      std::cout<<"GEM Rec Hit in [DetId] = ["<<rollId<<"] with BX = "<<recHit->BunchX()<<" and Global Position = "<<GEMGlobalPoint<<std::endl;
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TestGEMRecHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGEMRecHitAnalyzer);
