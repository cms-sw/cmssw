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
#include <Geometry/CommonTopologies/interface/GEMStripTopology.h>

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
  std::unique_ptr<TH1F> BX_RecHits_GE1in_Plus, BX_RecHits_GE1out_Plus, BX_RecHits_GE1in_Minus, BX_RecHits_GE1out_Minus;
  std::unique_ptr<TH1F> ST_RecHits_GE1in_Plus, ST_RecHits_GE1out_Plus, ST_RecHits_GE1in_Minus, ST_RecHits_GE1out_Minus;
  std::unique_ptr<TH1F> CL_RecHits_GE1in_Plus, CL_RecHits_GE1out_Plus, CL_RecHits_GE1in_Minus, CL_RecHits_GE1out_Minus;
  std::unique_ptr<TCanvas> BX_RecHits_GE1, ST_RecHits_GE1, CL_RecHits_GE1;

  std::vector<double> x_n1i, y_n1i, z_n1i, r_n1i, x_n1o, y_n1o, z_n1o, r_n1o;  // XYZR GE1 Minus in and out
  std::vector<double> x_p1i, y_p1i, z_p1i, r_p1i, x_p1o, y_p1o, z_p1o, r_p1o;  // XYZR GE1 Plus in and out
  std::vector<double> x_n1, y_n1, z_n1, r_n1, x_p1, y_p1, z_p1, r_p1;

  std::unique_ptr<TGraph> GE1in_Plus_XY_All, GE1in_Minus_XY_All, GE1out_Plus_XY_All, GE1out_Minus_XY_All,
      GE1out_Plus_YZ_All, GE1out_Minus_YZ_All, GE1in_Plus_YZ_All, GE1in_Minus_YZ_All, GE1_Plus_YZ_All, GE1_Minus_YZ_All;
  std::unique_ptr<TCanvas> Canvas_GE1_Plus_XY, Canvas_GE1_Minus_XY, Canvas_GE1_Plus_YZ, Canvas_GE1_Minus_YZ;
};

//
// constants, enums and typedefs
//
int n_bx = 11;
double n1_bx = -5.5, n2_bx = 5.5;
int n_st = 501;
double n1_st = 0, n2_st = 500;
int n_cl = 26;
double n1_cl = -0.5, n2_cl = 25.5;

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
  rootFileName = iConfig.getUntrackedParameter<std::string>("RootFileName");
  outputfile.reset(TFile::Open(rootFileName.c_str(), "RECREATE"));

  BX_RecHits_GE1in_Plus =
      std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1in_Plus", "BX_RecHits_GE1in_Plus", n_bx, n1_bx, n2_bx));
  BX_RecHits_GE1out_Plus =
      std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1out_Plus", "BX_RecHits_GE1out_Plus", n_bx, n1_bx, n2_bx));
  BX_RecHits_GE1in_Minus =
      std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1in_Minus", "BX_RecHits_GE1in_Minus", n_bx, n1_bx, n2_bx));
  BX_RecHits_GE1out_Minus =
      std::unique_ptr<TH1F>(new TH1F("BX_RecHits_GE1out_Minus", "BX_RecHits_GE1out_Minus", n_bx, n1_bx, n2_bx));

  ST_RecHits_GE1in_Plus =
      std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1in_Plus", "ST_RecHits_GE1in_Plus", n_st, n1_st, n2_st));
  ST_RecHits_GE1out_Plus =
      std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1out_Plus", "ST_RecHits_GE1out_Plus", n_st, n1_st, n2_st));
  ST_RecHits_GE1in_Minus =
      std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1in_Minus", "ST_RecHits_GE1in_Minus", n_st, n1_st, n2_st));
  ST_RecHits_GE1out_Minus =
      std::unique_ptr<TH1F>(new TH1F("ST_RecHits_GE1out_Minus", "ST_RecHits_GE1out_Minus", n_st, n1_st, n2_st));

  CL_RecHits_GE1in_Plus =
      std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1in_Plus", "CL_RecHits_GE1in_Plus", n_cl, n1_cl, n2_cl));
  CL_RecHits_GE1out_Plus =
      std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1out_Plus", "CL_RecHits_GE1out_Plus", n_cl, n1_cl, n2_cl));
  CL_RecHits_GE1in_Minus =
      std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1in_Minus", "CL_RecHits_GE1in_Minus", n_cl, n1_cl, n2_cl));
  CL_RecHits_GE1out_Minus =
      std::unique_ptr<TH1F>(new TH1F("CL_RecHits_GE1out_Minus", "CL_RecHits_GE1out_Minus", n_cl, n1_cl, n2_cl));
}

TestGEMRecHitAnalyzer::~TestGEMRecHitAnalyzer() {
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

  BX_RecHits_GE1 = std::unique_ptr<TCanvas>(new TCanvas("BX_RecHits_GE1", "BX_RecHits_GE1", 800, 600));
  ST_RecHits_GE1 = std::unique_ptr<TCanvas>(new TCanvas("ST_RecHits_GE1", "ST_RecHits_GE1", 800, 600));
  CL_RecHits_GE1 = std::unique_ptr<TCanvas>(new TCanvas("CL_RecHits_GE1", "CL_RecHits_GE1", 800, 600));

  BX_RecHits_GE1->cd();
  BX_RecHits_GE1->Divide(2, 2);
  BX_RecHits_GE1->cd(1);
  BX_RecHits_GE1in_Plus->Draw();
  BX_RecHits_GE1in_Plus->GetXaxis()->SetTitle("BX [-]");
  BX_RecHits_GE1in_Plus->GetYaxis()->SetTitle("entries [-]");
  BX_RecHits_GE1in_Plus->SetTitle("GE+1in RecHits");
  BX_RecHits_GE1->cd(2);
  BX_RecHits_GE1out_Plus->Draw();
  BX_RecHits_GE1out_Plus->GetXaxis()->SetTitle("BX [-]");
  BX_RecHits_GE1out_Plus->GetYaxis()->SetTitle("entries [-]");
  BX_RecHits_GE1out_Plus->SetTitle("GE+1out RecHits");
  BX_RecHits_GE1->cd(3);
  BX_RecHits_GE1in_Minus->Draw();
  BX_RecHits_GE1in_Minus->GetXaxis()->SetTitle("BX [-]");
  BX_RecHits_GE1in_Minus->GetYaxis()->SetTitle("entries [-]");
  BX_RecHits_GE1in_Minus->SetTitle("GE-1in RecHits");
  BX_RecHits_GE1->cd(4);
  BX_RecHits_GE1out_Minus->Draw();
  BX_RecHits_GE1out_Minus->GetXaxis()->SetTitle("BX [-]");
  BX_RecHits_GE1out_Minus->GetYaxis()->SetTitle("entries [-]");
  BX_RecHits_GE1out_Minus->SetTitle("GE-1out RecHits");

  ST_RecHits_GE1->cd();
  ST_RecHits_GE1->Divide(2, 2);
  ST_RecHits_GE1->cd(1);
  ST_RecHits_GE1in_Plus->Draw();
  ST_RecHits_GE1in_Plus->GetXaxis()->SetTitle("First Strip of Cluster [-]");
  ST_RecHits_GE1in_Plus->GetYaxis()->SetTitle("entries [-]");
  ST_RecHits_GE1in_Plus->SetTitle("GE+1in RecHits");
  ST_RecHits_GE1->cd(2);
  ST_RecHits_GE1out_Plus->Draw();
  ST_RecHits_GE1out_Plus->GetXaxis()->SetTitle("First Strip of Cluster [-]");
  ST_RecHits_GE1out_Plus->GetYaxis()->SetTitle("entries [-]");
  ST_RecHits_GE1out_Plus->SetTitle("GE+1out RecHits");
  ST_RecHits_GE1->cd(3);
  ST_RecHits_GE1in_Minus->Draw();
  ST_RecHits_GE1in_Minus->GetXaxis()->SetTitle("First Strip of Cluster [-]");
  ST_RecHits_GE1in_Minus->GetYaxis()->SetTitle("entries [-]");
  ST_RecHits_GE1in_Minus->SetTitle("GE-1in RecHits");
  ST_RecHits_GE1->cd(4);
  ST_RecHits_GE1out_Minus->Draw();
  ST_RecHits_GE1out_Minus->GetXaxis()->SetTitle("First Strip of Cluster [-]");
  ST_RecHits_GE1out_Minus->GetYaxis()->SetTitle("entries [-]");
  ST_RecHits_GE1out_Minus->SetTitle("GE-1out RecHits");

  CL_RecHits_GE1->cd();
  CL_RecHits_GE1->Divide(2, 2);
  CL_RecHits_GE1->cd(1);
  CL_RecHits_GE1in_Plus->Draw();
  CL_RecHits_GE1in_Plus->GetXaxis()->SetTitle("Clustersize [-]");
  CL_RecHits_GE1in_Plus->GetYaxis()->SetTitle("entries [-]");
  CL_RecHits_GE1in_Plus->SetTitle("GE+1in RecHits");
  CL_RecHits_GE1->cd(2);
  CL_RecHits_GE1out_Plus->Draw();
  CL_RecHits_GE1out_Plus->GetXaxis()->SetTitle("Clustersize [-]");
  CL_RecHits_GE1out_Plus->GetYaxis()->SetTitle("entries [-]");
  CL_RecHits_GE1out_Plus->SetTitle("GE+1out RecHits");
  CL_RecHits_GE1->cd(3);
  CL_RecHits_GE1in_Minus->Draw();
  CL_RecHits_GE1in_Minus->GetXaxis()->SetTitle("Clustersize [-]");
  CL_RecHits_GE1in_Minus->GetYaxis()->SetTitle("entries [-]");
  CL_RecHits_GE1in_Minus->SetTitle("GE-1in RecHits");
  CL_RecHits_GE1->cd(4);
  CL_RecHits_GE1out_Minus->Draw();
  CL_RecHits_GE1out_Minus->GetXaxis()->SetTitle("Clustersize [-]");
  CL_RecHits_GE1out_Minus->GetYaxis()->SetTitle("entries [-]");
  CL_RecHits_GE1out_Minus->SetTitle("GE-1out RecHits");

  const int n_n1i = x_n1i.size();
  double x_an1i[n_n1i];
  double y_an1i[n_n1i];
  double z_an1i[n_n1i];
  double r_an1i[n_n1i];
  const int n_n1o = x_n1o.size();
  double x_an1o[n_n1o];
  double y_an1o[n_n1o];
  double z_an1o[n_n1o];
  double r_an1o[n_n1o];
  const int n_p1i = x_p1i.size();
  double x_ap1i[n_p1i];
  double y_ap1i[n_p1i];
  double z_ap1i[n_p1i];
  double r_ap1i[n_p1i];
  const int n_p1o = x_p1o.size();
  double x_ap1o[n_p1o];
  double y_ap1o[n_p1o];
  double z_ap1o[n_p1o];
  double r_ap1o[n_p1o];
  const int n_n1 = x_n1.size(); /*double x_an1[n_n1]; double y_an1[n_n1];*/
  double z_an1[n_n1];
  double r_an1[n_n1];
  const int n_p1 = x_p1.size(); /*double x_ap1[n_p1]; double y_ap1[n_p1];*/
  double z_ap1[n_p1];
  double r_ap1[n_p1];

  for (int i = 0; i < n_n1i; ++i) {
    x_an1i[i] = x_n1i[i];
    y_an1i[i] = y_n1i[i];
    z_an1i[i] = z_n1i[i];
    r_an1i[i] = r_n1i[i];
  }
  for (int i = 0; i < n_n1o; ++i) {
    x_an1o[i] = x_n1o[i];
    y_an1o[i] = y_n1o[i];
    z_an1o[i] = z_n1o[i];
    r_an1o[i] = r_n1o[i];
  }
  for (int i = 0; i < n_p1i; ++i) {
    x_ap1i[i] = x_p1i[i];
    y_ap1i[i] = y_p1i[i];
    z_ap1i[i] = z_p1i[i];
    r_ap1i[i] = r_p1i[i];
  }
  for (int i = 0; i < n_p1o; ++i) {
    x_ap1o[i] = x_p1o[i];
    y_ap1o[i] = y_p1o[i];
    z_ap1o[i] = z_p1o[i];
    r_ap1o[i] = r_p1o[i];
  }
  for (int i = 0; i < n_n1; ++i) { /*x_an1[i] = x_n1[i]; y_an1[i] = y_n1[i];*/
    z_an1[i] = z_n1[i];
    r_an1[i] = r_n1[i];
  }
  for (int i = 0; i < n_p1; ++i) { /*x_ap1[i] = x_p1[i]; y_ap1[i] = y_p1[i];*/
    z_ap1[i] = z_p1[i];
    r_ap1[i] = r_p1[i];
  }

  GE1in_Minus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_n1i, x_an1i, y_an1i));
  std::cout << "GE-1in All SimHits: " << n_n1i << std::endl;
  GE1out_Minus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_n1o, x_an1o, y_an1o));
  std::cout << "GE-1out All SimHits: " << n_n1o << std::endl;
  GE1in_Plus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_p1i, x_ap1i, y_ap1i));
  std::cout << "GE+1in All SimHits: " << n_p1i << std::endl;
  GE1out_Plus_XY_All = std::unique_ptr<TGraph>(new TGraph(n_p1o, x_ap1o, y_ap1o));
  std::cout << "GE+1out All SimHits: " << n_p1o << std::endl;

  GE1in_Minus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_n1i, z_an1i, r_an1i));
  GE1out_Minus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_n1o, z_an1o, r_an1o));
  GE1in_Plus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_p1i, z_ap1i, r_ap1i));
  GE1out_Plus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_p1o, z_ap1o, r_ap1o));

  GE1_Minus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_n1, z_an1, r_an1));
  GE1_Plus_YZ_All = std::unique_ptr<TGraph>(new TGraph(n_p1, z_ap1, r_ap1));

  Canvas_GE1_Plus_XY = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Plus_XY", "Canvas_GE1_Plus_XY", 800, 600));
  Canvas_GE1_Minus_XY = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Minus_XY", "Canvas_GE1_Minus_XY", 800, 600));
  Canvas_GE1_Plus_YZ = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Plus_YZ", "Canvas_GE1_Plus_YZ", 600, 800));
  Canvas_GE1_Minus_YZ = std::unique_ptr<TCanvas>(new TCanvas("Canvas_GE1_Minus_YZ", "Canvas_GE1_Minus_YZ", 600, 800));

  // XY and RZ Graphs
  Canvas_GE1_Plus_XY->cd();
  GE1in_Plus_XY_All->SetMarkerStyle(mar1);
  GE1in_Plus_XY_All->SetMarkerColor(col1);
  GE1in_Plus_XY_All->Draw("AP");
  GE1out_Plus_XY_All->SetMarkerStyle(mar2);
  GE1out_Plus_XY_All->SetMarkerColor(col2);
  GE1out_Plus_XY_All->Draw("Psame");
  GE1in_Plus_XY_All->GetXaxis()->SetTitle("X [cm]");
  GE1in_Plus_XY_All->GetYaxis()->SetTitle("Y [cm]");
  GE1in_Plus_XY_All->SetTitle("GE+1 RecHits");

  Canvas_GE1_Minus_XY->cd();
  GE1in_Minus_XY_All->SetMarkerStyle(mar1);
  GE1in_Minus_XY_All->SetMarkerColor(col1);
  GE1in_Minus_XY_All->Draw("AP");
  GE1out_Minus_XY_All->SetMarkerStyle(mar2);
  GE1out_Minus_XY_All->SetMarkerColor(col2);
  GE1out_Minus_XY_All->Draw("Psame");
  GE1in_Minus_XY_All->GetXaxis()->SetTitle("X [cm]");
  GE1in_Minus_XY_All->GetYaxis()->SetTitle("Y [cm]");
  GE1in_Minus_XY_All->SetTitle("GE-1 RecHits");

  Canvas_GE1_Plus_YZ->cd();
  GE1_Plus_YZ_All->SetMarkerStyle(0);
  GE1_Plus_YZ_All->SetMarkerColor(0);
  GE1_Plus_YZ_All->Draw("AP");
  GE1_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]");
  GE1_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]");
  GE1_Plus_YZ_All->SetTitle("GE+1 RecHits");
  GE1in_Plus_YZ_All->SetMarkerStyle(mar1);
  GE1in_Plus_YZ_All->SetMarkerColor(col1);
  GE1in_Plus_YZ_All->Draw("Psame");
  GE1out_Plus_YZ_All->SetMarkerStyle(mar2);
  GE1out_Plus_YZ_All->SetMarkerColor(col2);
  GE1out_Plus_YZ_All->Draw("Psame");
  // GE1in_Plus_YZ_All->GetXaxis()->SetTitle("Z [cm]");    GE1in_Plus_YZ_All->GetYaxis()->SetTitle("R [cm]");     GE1in_Plus_YZ_All->SetTitle("GE1 Plus RecHits");
  // GE1in_Plus_YZ_All->GetXaxis()->SetRangeUser(563,573);
  // TLatex latex1; latex1.SetNDC();  latex1.SetTextAlign(23);
  // latex1.SetTextSize(0.03); latex1.DrawLatex(0.175,0.50,"#color[15]{RE+1/2}"); latex1.DrawLatex(0.325,0.50,"#color[15]{RE+2/2}"); latex1.DrawLatex(0.675,0.50,"#color[15]{RE+3/2}"); latex1.DrawLatex(0.825,0.50,"#color[15]{RE+4/2}");
  // latex1.SetTextSize(0.03); latex1.DrawLatex(0.175,0.875,"#color[15]{RE+1/3}"); latex1.DrawLatex(0.325,0.875,"#color[15]{RE+2/3}"); latex1.DrawLatex(0.675,0.875,"#color[15]{RE+3/3}"); latex1.DrawLatex(0.825,0.875,"#color[15]{RE+4/3}");

  Canvas_GE1_Minus_YZ->cd();
  GE1_Minus_YZ_All->SetMarkerStyle(0);
  GE1_Minus_YZ_All->SetMarkerColor(0);
  GE1_Minus_YZ_All->Draw("AP");
  GE1_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]");
  GE1_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]");
  GE1_Minus_YZ_All->SetTitle("GE-1 RecHits");
  GE1in_Minus_YZ_All->SetMarkerStyle(mar1);
  GE1in_Minus_YZ_All->SetMarkerColor(col1);
  GE1in_Minus_YZ_All->Draw("Psame");
  GE1out_Minus_YZ_All->SetMarkerStyle(mar2);
  GE1out_Minus_YZ_All->SetMarkerColor(col2);
  GE1out_Minus_YZ_All->Draw("Psame");
  // GE1in_Minus_YZ_All->GetXaxis()->SetTitle("Z [cm]");   GE1in_Minus_YZ_All->GetYaxis()->SetTitle("R [cm]");    GE1in_Minus_YZ_All->SetTitle("GE Minus RecHits");
  // GE1in_Minus_YZ_All->GetXaxis()->SetRangeUser(-573, -563);
  // TLatex latex2; latex2.SetNDC();  latex2.SetTextAlign(23);
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.175,0.50,"#color[15]{RE-4/2}"); latex2.DrawLatex(0.325,0.50,"#color[15]{RE-3/2}"); latex2.DrawLatex(0.675,0.50,"#color[15]{RE-2/2}"); latex2.DrawLatex(0.825,0.50,"#color[15]{RE-1/2}");
  // latex2.SetTextSize(0.03); latex2.DrawLatex(0.175,0.875,"#color[15]{RE-4/3}"); latex2.DrawLatex(0.325,0.875,"#color[15]{RE-3/3}"); latex2.DrawLatex(0.675,0.875,"#color[15]{RE-2/3}"); latex2.DrawLatex(0.825,0.875,"#color[15]{RE-1/3}");

  BX_RecHits_GE1->Write();
  ST_RecHits_GE1->Write();
  CL_RecHits_GE1->Write();
  Canvas_GE1_Plus_XY->Write();
  Canvas_GE1_Minus_XY->Write();
  Canvas_GE1_Plus_YZ->Write();
  Canvas_GE1_Minus_YZ->Write();
}

//
// member functions
//

// ------------ method called for each event  ------------
void TestGEMRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(gemGeom);

  // ================
  // GEM recHits
  // ================
  edm::Handle<GEMRecHitCollection> gemRecHits;
  iEvent.getByToken(GEMRecHit_Token, gemRecHits);

  // count the number of GEM rechits
  int nGEM = 0;
  GEMRecHitCollection::const_iterator recHit;
  for (recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    nGEM++;
  }

  // std::cout<<"The Number of RecHits is "<<nGEM<<std::endl;
  for (recHit = gemRecHits->begin(); recHit != gemRecHits->end(); recHit++) {
    GEMDetId rollId = (GEMDetId)(*recHit).gemId();
    LocalPoint recHitPos = recHit->localPosition();
    const GEMEtaPartition* rollasociated = gemGeom->etaPartition(rollId);
    const BoundPlane& GEMSurface = rollasociated->surface();
    GlobalPoint GEMGlobalPoint = GEMSurface.toGlobal(recHitPos);

    int region = rollId.region();
    int station = rollId.station();
    // int ring    = rollId.ring();
    int layer = rollId.layer();
    // int etapart = rollId.roll();
    // int chamber = rollId.chamber();

    std::cout << "GEM Rec Hit in [DetId] = [" << rollId << "] with BX = " << recHit->BunchX()
              << " and Global Position = " << GEMGlobalPoint << std::endl;

    int bx = recHit->BunchX();
    int cl = recHit->clusterSize();
    int st = recHit->firstClusterStrip();

    // Positive Endcap
    if (region == 1 && station == 1) {
      x_p1.push_back(GEMGlobalPoint.x());
      y_p1.push_back(GEMGlobalPoint.y());
      z_p1.push_back(GEMGlobalPoint.z());
      r_p1.push_back(sqrt(pow(GEMGlobalPoint.x(), 2) + pow(GEMGlobalPoint.y(), 2)));
      // Layers
      if (layer == 1) {
        BX_RecHits_GE1in_Plus->Fill(bx);
        ST_RecHits_GE1in_Plus->Fill(st);
        CL_RecHits_GE1in_Plus->Fill(cl);
        x_p1i.push_back(GEMGlobalPoint.x());
        y_p1i.push_back(GEMGlobalPoint.y());
        z_p1i.push_back(GEMGlobalPoint.z());
        r_p1i.push_back(sqrt(pow(GEMGlobalPoint.x(), 2) + pow(GEMGlobalPoint.y(), 2)));
      }
      if (layer == 2) {
        BX_RecHits_GE1out_Plus->Fill(bx);
        ST_RecHits_GE1out_Plus->Fill(st);
        CL_RecHits_GE1out_Plus->Fill(cl);
        x_p1o.push_back(GEMGlobalPoint.x());
        y_p1o.push_back(GEMGlobalPoint.y());
        z_p1o.push_back(GEMGlobalPoint.z());
        r_p1o.push_back(sqrt(pow(GEMGlobalPoint.x(), 2) + pow(GEMGlobalPoint.y(), 2)));
      }
    }
    // Negative Endcap
    if (region == -1 && station == 1) {
      x_n1.push_back(GEMGlobalPoint.x());
      y_n1.push_back(GEMGlobalPoint.y());
      z_n1.push_back(GEMGlobalPoint.z());
      r_n1.push_back(sqrt(pow(GEMGlobalPoint.x(), 2) + pow(GEMGlobalPoint.y(), 2)));
      // Layers
      if (layer == 1) {
        BX_RecHits_GE1in_Minus->Fill(bx);
        ST_RecHits_GE1in_Minus->Fill(st);
        CL_RecHits_GE1in_Minus->Fill(cl);
        x_n1i.push_back(GEMGlobalPoint.x());
        y_n1i.push_back(GEMGlobalPoint.y());
        z_n1i.push_back(GEMGlobalPoint.z());
        r_n1i.push_back(sqrt(pow(GEMGlobalPoint.x(), 2) + pow(GEMGlobalPoint.y(), 2)));
      }
      if (layer == 2) {
        BX_RecHits_GE1out_Minus->Fill(bx);
        ST_RecHits_GE1out_Minus->Fill(st);
        CL_RecHits_GE1out_Minus->Fill(cl);
        x_n1o.push_back(GEMGlobalPoint.x());
        y_n1o.push_back(GEMGlobalPoint.y());
        z_n1o.push_back(GEMGlobalPoint.z());
        r_n1o.push_back(sqrt(pow(GEMGlobalPoint.x(), 2) + pow(GEMGlobalPoint.y(), 2)));
      }
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TestGEMRecHitAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestGEMRecHitAnalyzer);
