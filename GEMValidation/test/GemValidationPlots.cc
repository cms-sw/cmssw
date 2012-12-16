/*  Script to produce validation plots.
  Run in terminal, type:
  root -l
  .L GemValidation.cc
  main()
  
  You will be prompted to type a production identifier, e.g. "SingleMuPt40_20121128_Pilot".
  The program will look for two files in the *working* directory:
  - SingleMuPt40_20121128_Pilot_SIM.root
  - SingleMuPt40_20121128_Pilot_DIGI.root
  These files are of course the output of the *GEMAnalyzers*
  - GEMSimHitAnalyzer
  - GEMDigiAnalyzer
  In case one is or both are missing, the program will terminate automatically.
  Otherwise an output directory is created in the *working* directory, e.g. GemValidation_SingleMuPt40_20121128_Pilot/.
  Plots (.pdf format) are stored in the output directory. 
  
  FIXME:
  - include check for existing directories
  - include path to ROOT files not in working directory

  Contact sven.dildick@cern.ch for more information
 */

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TPad.h"
#include "TStyle.h"
#include "TString.h"
#include "TAxis.h"
#include "TArrayD.h"

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string>
#include <sstream>

// convert to string
template < typename T > std::string to_string( T const& value )
{
  std::stringstream sstr;
  sstr << value;
  return sstr.str();
}

void setEtaBinLabels(TH1D* h)
{
  h->GetXaxis()->SetBinLabel(1,"-11");
  h->GetXaxis()->SetBinLabel(2,"-12");
  h->GetXaxis()->SetBinLabel(3,"-13");
  h->GetXaxis()->SetBinLabel(4,"-14");
  h->GetXaxis()->SetBinLabel(5,"-15");
  h->GetXaxis()->SetBinLabel(6,"-16");
  h->GetXaxis()->SetBinLabel(7,"-21");
  h->GetXaxis()->SetBinLabel(8,"-22");
  h->GetXaxis()->SetBinLabel(9,"-23");
  h->GetXaxis()->SetBinLabel(10,"-24");
  h->GetXaxis()->SetBinLabel(11,"-25");
  h->GetXaxis()->SetBinLabel(12,"-26");
  h->GetXaxis()->SetBinLabel(13,"11");
  h->GetXaxis()->SetBinLabel(14,"12");
  h->GetXaxis()->SetBinLabel(15,"13");
  h->GetXaxis()->SetBinLabel(16,"14");
  h->GetXaxis()->SetBinLabel(17,"15");
  h->GetXaxis()->SetBinLabel(18,"16");
  h->GetXaxis()->SetBinLabel(19,"21");
  h->GetXaxis()->SetBinLabel(20,"22");
  h->GetXaxis()->SetBinLabel(21,"23");
  h->GetXaxis()->SetBinLabel(22,"24");
  h->GetXaxis()->SetBinLabel(23,"25");
  h->GetXaxis()->SetBinLabel(24,"26");
}

int main(int argc, char* argv[] )
{
  // To make the plot a bit prettier...
  gStyle->SetPalette(1);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadTopMargin(0.12);
  // gStyle->SetPadRightMargin(0.12);
  gStyle->SetPadRightMargin(0.4);
  gStyle->SetPadBottomMargin(.12);
  gStyle->SetPadLeftMargin(.12);
  gStyle->SetTitleSize(.05,"XYZ");
  gStyle->SetTitleFillColor(kWhite);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatColor(kWhite);
  gStyle->SetStatBorderSize(1);
  gStyle->SetOptStat( 1111 );
  gStyle->SetOptFit( 0 );
  gStyle->SetMarkerStyle(8);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetNumberContours(50);

  // Steering  
  std::cout << ">>> Enter MC production identifier:" << std::endl;
  std::string identifier = "gem_ana";
  // std::cin >> identifier;

  // use ROOT files in working directory
  const TString simHitFileName( identifier + "_SIM.root" );
  const TString digiFileName( identifier + "_DIGI.root" );
  std::cout << ">>> Using input files: " << std::endl
	    << "\t" << simHitFileName << std::endl
	    << "\t" << digiFileName << std::endl;

  // Check for availability ROOT files
  TFile *simHitFile = new TFile( simHitFileName, "READ" );
  TFile *digiFile = new TFile( digiFileName, "READ"  );
  if (!simHitFile || !digiFile ){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TFile: " << simHitFileName 
	      << "\n>>> Error in GemValidationPlots::main() No such TFile: " << digiFileName << std::endl;
    exit(1);
  }

  // create output directory in the working directory
  const TString outputDir( "validationPlots_" + identifier + "/");
  std::cout << ">>> Creating output directory: " << outputDir << std::endl;
  struct stat sb;
  system( "mkdir " + outputDir );

  // extension for the plots [.pdf/.png]
  // TString ext( ".png");
  TString ext( ".pdf");

  /////////////////////////////
  // SIMHIT VALIDATION PLOTS //
  /////////////////////////////

  std::cout << ">>> Opening TFile: " << simHitFileName << std::endl;  
  std::cout << ">>> Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
  TDirectory * dir = (TDirectory*)simHitFile->Get("gemSimHitAnalyzer");
  if (!dir){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TDirectory: gemSimHitAnalyzer" << std::endl;
    exit(1);
  }

  std::cout << ">>> Reading TTree: GEMSimHits" << std::endl;
  TTree* tree = (TTree*) dir->Get("GEMSimHits");
  if (!tree){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TTree: GEMSimHits" << std::endl;
    exit(1);
  }
  
  //--------------------//
  // XY occupancy plots //
  //--------------------//

  //--------Muon--------//
  
  TCanvas* c = new TCanvas("c","c",1000,600);
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1&&particleType==13");
  TH2D *hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Muon SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region-1_layer1_muon_simhit" + ext );

  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2&&particleType==13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Muon SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region-1_layer2_muon_simhit" + ext);

  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==1&&particleType==13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Muon SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region1_layer1_muon_simhit" + ext);

  c->Clear();	
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==2&&particleType==13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Muon SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalxy_region1_layer2_muon_simhit" + ext);
  
  //--------Non muon--------//

  c->Clear();	
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1&&particleType!=13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region-1_layer1_nonmuon_simhit" + ext );
  
  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2&&particleType!=13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Non muon SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region-1_layer2_nonmuon_simhit" + ext);

  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==1&&particleType!=13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Non muon SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region1_layer1_nonmuon_simhit" + ext);

  c->Clear();	
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==2&&particleType!=13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Non muon SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalxy_region1_layer2_nonmuon_simhit" + ext);

  //--------All--------//
  
  c->Clear();	
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("All particle SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region-1_layer1_all_simhit" + ext );
  
  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("All particle SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region-1_layer2_all_simhit" + ext);

  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260, 100,-260,260)","region==1&&layer==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("All particle SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs( outputDir + "globalxy_region1_layer1_all_simhit" + ext);

  c->Clear();	
  tree->Draw("globalY:globalX>>hh(100,-260,260, 100,-260,260)","region==1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("All particle SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalxy_region1_layer2_all_simhit" + ext);

  //--------------------//
  // ZR occupancy plots //
  //--------------------//

  //--------Muon--------//

  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1&&particleType==13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Muon SimHit occupancy: region2;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalzr_region-1_muon_simhit" + ext);

  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1&&particleType==13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Muon SimHit occupancy: region1;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalzr_region1_muon_simhit" + ext);

  //--------Non muon--------//

  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1&&particleType!=13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Non muon SimHit occupancy: region2;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalzr_region-1_nonmuon_simhit" + ext);

  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1&&particleType!=13");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Non muon SimHit occupancy: region1;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalzr_region1_nonmuon_simhit" + ext);

  //--------All--------//

  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("All particle SimHit occupancy: region2;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalzr_region-1_all_simhit" + ext);

  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("All particle SimHit occupancy: region1;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(outputDir + "globalzr_region1_all_simhit" + ext);

  //--------------------//
  // timeOfFlight plots //
  //--------------------//

  //--------Muon--------//

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1&&particleType==13");
  TH1D* h = (TH1D*)gDirectory->Get("h");
  TString title( "Muon SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns" );
  h->SetTitle( title );
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_muon_simhit" + ext);

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2&&particleType==13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Muon SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_muon_simhit" + ext);       

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1&&particleType==13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Muon SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region1_layer1_muon_simhit" + ext);

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2&&particleType==13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Muon SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region1_layer2_muon_simhit" + ext);

  //--------Non muon--------//

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1&&particleType!=13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Non muon SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_nonmuon_simhit" + ext);

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2&&particleType!=13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Non muon SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_nonmuon_simhit" + ext);       

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1&&particleType!=13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Non muon SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region1_layer1_nonmuon_simhit" + ext);

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2&&particleType!=13");
  h = (TH1D*)gDirectory->Get("h");
  title = "Non muon SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region1_layer2_nonmuon_simhit" + ext);

  //--------All--------//

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  title =  "All particle SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_all_simhit" + ext);

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  title = "All particle SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_all_simhit" + ext);       

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  title = "All particle SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region1_layer1_all_simhit" + ext);

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  title = "All particle SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(1)) + " ns";
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "timeOfFlight_region1_layer2_all_simhit" + ext);

  //--------------------//
  // energy loss plots  //
  //--------------------//
  
  TH1D* muonEnergyLoss = new TH1D("muonEnergyLoss","",60,0.,6000.);
  TH1D* nonMuonEnergyLoss = new TH1D("nonMuonEnergyLoss","",60,0.,6000.);
  TH1D* allEnergyLoss = new TH1D("allEnergyLoss","",60,0.,6000.);
  int particleType=0;
  Float_t energyLoss=0;
  TBranch *b_particleType;  
  TBranch *b_energyLoss;
  tree->SetBranchAddress("energyLoss", &energyLoss, &b_energyLoss);  
  tree->SetBranchAddress("particleType", &particleType, &b_particleType);
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<tree->GetEntriesFast();jentry++) {
    Long64_t ientry = tree->LoadTree(jentry);
    if (ientry < 0) break;
    nb = tree->GetEntry(jentry);   
    nbytes += nb;
    if ( particleType==13 ) muonEnergyLoss->Fill( energyLoss*1.e9 );
    if ( particleType!=13 ) nonMuonEnergyLoss->Fill( energyLoss*1.e9 );
    allEnergyLoss->Fill( energyLoss*1.e9 );
  }
  c->Clear();
  title = "Muon energy loss;Energy loss [eV];entries/" + to_string(h->GetBinWidth(1)) + " eV";
  muonEnergyLoss->SetTitle( title );
  muonEnergyLoss->Draw("");        
  c->SaveAs(outputDir + "energyLoss_muon_simhit" + ext);

  c->Clear();
  title = "Non muon energy loss;Energy loss [eV];entries/" + to_string(h->GetBinWidth(1)) + " eV";
  nonMuonEnergyLoss->SetTitle( title );
  nonMuonEnergyLoss->Draw("");        
  c->SaveAs(outputDir + "energyLoss_nonmuon_simhit" + ext);

  c->Clear();
  title = "All particle energy loss;Energy loss [eV];entries/" +  to_string(h->GetBinWidth(1)) + " eV";
  allEnergyLoss->SetTitle( title );
  allEnergyLoss->Draw("");        
  c->SaveAs(outputDir + "energyLoss_all_simhit" + ext);

  //--------------------//
  //   momentum plots   //
  //--------------------//
  
  c->Clear();
  tree->Draw("pabs>>h(50,0.,200.)");
  h = (TH1D*)gDirectory->Get("h");
  gPad->SetLogx(0);
  title = "SimHits absolute momentum;Momentum [GeV];entries/" +  to_string(h->GetBinWidth(1)) + " GeV"; // check units here
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "momentum_all_simhit" + ext);

  c->Clear();
  tree->Draw("pabs>>h(50,0.,200.)","particleType==13");
  h = (TH1D*)gDirectory->Get("h");
  // gPad->SetLogx();
  title = "SimHits absolute momentum;Momentum [GeV];entries/" +  to_string(h->GetBinWidth(1)) + " GeV"; // check units here
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "momentum_muon_simhit" + ext);

  c->Clear();
  tree->Draw("pabs>>h(50,0.,200.)","particleType!=13");
  h = (TH1D*)gDirectory->Get("h");
  // gPad->SetLogx();
  title = "SimHits absolute momentum;Momentum [GeV];entries/" +  to_string(h->GetBinWidth(1)) + " GeV"; // check units here
  h->SetTitle( title );       
  h->Draw("");        
  c->SaveAs(outputDir + "momentum_nonmuon_simhit" + ext);

  //--------------------//
  //    pdg ID plots    //
  //--------------------//

  c->Clear();
  gPad->SetLogx(0);
  tree->Draw("particleType>>h(25,0.,25.)","particleType>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "SimHit PDG Id;PDG Id;entries"  );       
  h->Draw("");        
  c->SaveAs(outputDir + "particleType_simhit" + ext);

  //--------------------//
  // eta occupancy plot //
  //--------------------//

  TH1D* muonEtaOccupancy = new TH1D("muonEtaOccupancy","Muon globalEta",24,1.,25.);
  TH1D* nonMuonEtaOccupancy = new TH1D("nonMuonEtaOccupancy","Non muon globalEta",24,1.,25.);
  TH1D* allEtaOccupancy = new TH1D("allEtaOccupancy","All particle globalEta",24,1.,25.);
  int region=0;
  int layer=0;
  int roll=0;  
  TBranch *b_region;
  TBranch *b_layer;
  TBranch *b_roll;
  tree->SetBranchAddress("region", &region, &b_region);
  tree->SetBranchAddress("layer", &layer, &b_layer);
  tree->SetBranchAddress("roll", &roll, &b_roll);
  nbytes = 0;
  nb = 0;
  for (Long64_t jentry=0; jentry<tree->GetEntriesFast();jentry++) {
    Long64_t ientry = tree->LoadTree(jentry);
    if (ientry < 0) break;
    nb = tree->GetEntry(jentry);   
    nbytes += nb;
    if (particleType==13) muonEtaOccupancy->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
    if (particleType!=13) nonMuonEtaOccupancy->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
                          allEtaOccupancy->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
  }    
  c->Clear();  
  setEtaBinLabels(muonEtaOccupancy);
  muonEtaOccupancy->Draw("");        
  c->SaveAs(outputDir + "globalEta_muon_simhit" + ext);

  c->Clear();  
  setEtaBinLabels(nonMuonEtaOccupancy);
  nonMuonEtaOccupancy->Draw("");        
  c->SaveAs(outputDir + "globalEta_nonmuon_simhit" + ext);

  c->Clear();  
  setEtaBinLabels(allEtaOccupancy);
  allEtaOccupancy->Draw("");        
  c->SaveAs(outputDir + "globalEta_all_simhit" + ext);

    
  delete hh;
  delete h;
  delete c;
  delete tree;
  delete dir;
  

  ///////////////////////////
  // DIGI VALIDATION PLOTS //
  ///////////////////////////

  std::cout << ">>> Opening TFile: " << digiFileName << std::endl;  
  std::cout << ">>> Opening TDirectory: gemDigiAnalyzer" << std::endl;  
  dir = (TDirectory*)digiFile->Get("gemDigiAnalyzer");
  if (!dir){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TDirectory: gemDigiAnalyzer" << std::endl;
    exit(1);
  }
  
  std::cout << ">>> Reading TTree: GemSimDigiTree" << std::endl;
  tree = (TTree*) dir->Get("GEMDigiTree");
  if (!tree){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TTree: GemDigiTree" << std::endl;
    exit(1);
  } 

  //--------------------//
  // XY occupancy plots //
  //--------------------//
  
  c = new TCanvas("c","c",1000,600);
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==1");
  hh = (TH2D*)gDirectory->Get("hh");  
  hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "globalxy_region-1_layer1_all_digi" + ext);

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "globalxy_region-1_layer2_all_digi" + ext);

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "globalxy_region1_layer1_all_digi" + ext);

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");	
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "globalxy_region1_layer2_all_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==1&&particleType==13");
  // TH2D *hh = (TH2D*)gDirectory->Get("hh");  
  // hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region-1_layer1_muon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==2&&particleType==13");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region-1_layer2_muon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==1&&particleType==13");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region1_layer1_muon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==2&&particleType==13");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");	
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region1_layer2_muon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==1&&particleType!=13");
  // TH2D *hh = (TH2D*)gDirectory->Get("hh");  
  // hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region-1_layer1_nonmuon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==2&&particleType!=13");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region-1_layer2_nonmuon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==1&&particleType!=13");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region1_layer1_nonmuon_digi" + ext);

  // c->Clear();
  // tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==2&&particleType!=13");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");	
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalxy_region1_layer2_nonmuon_digi" + ext);


  //--------------------//
  // ZR occupancy plots //
  //--------------------//

  // c->Clear();	
  // tree->Draw("g_r:g_z>>hh(30,-568,-565,22,130,240)","region==-1&&layer==1");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region-1, layer1; globalZ [cm]; globalR [cm]");	
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalzr_region-1_layer1_digi" + ext);

  // c->Clear();	
  // tree->Draw("g_r:g_z>>hh(30,-572.25,-569.25,22,130,240)","region==-1&&layer==2");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region-1, layer2; globalZ [cm]; globalR [cm]");	
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalzr_region-1_layer2_digi" + ext);
	  
  // c->Clear();	
  // tree->Draw("g_r:g_z>>hh(30,565,568,22,130,240)","region==1&&layer==1");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer1; globalZ [cm]; globalR [cm]");	
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalzr_region1_layer1_digi" + ext);
	
  // c->Clear();		
  // tree->Draw("g_r:g_z>>hh(30,569.25,572.25,22,130,240)","region==1&&layer==2");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer2; globalZ [cm]; globalR [cm]");	
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "globalzr_region1_layer2_digi" + ext);

  c->Clear();		
  tree->Draw("g_r:g_z>>hh(200,-573,-564,55,130,240)","region==-1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "globalzr_region-1_digi" + ext);

  c->Clear();		
  tree->Draw("g_r:g_z>>hh(200,564,573,55,130,240)","region==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "globalzr_region1_digi" + ext);


  //--------------------//
  //   PhiStrip plots   //
  //--------------------//

  // c->Clear();		
  // tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400,)","region==-1&&layer==1");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region-1, layer1; phi [rad]; strip");		
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "phiStrip_region-1_layer1_digi" + ext);

  // c->Clear();		
  // tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400)","region==-1&&layer==2");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region-1, layer2; phi [rad]; strip");		
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "phiStrip_region-1_layer2_digi" + ext);

  // c->Clear();		
  // tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400)","region==1&&layer==1");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer1; phi [rad]; strip");		
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "phiStrip_region1_layer1_digi" + ext);

  // c->Clear();		
  // tree->Draw("strip:g_phi>>hh(140,-3.5,3.5,100,0,400)","region==1&&layer==2");
  // hh = (TH2D*)gDirectory->Get("hh");
  // hh->SetTitle("Digi occupancy: region1, layer2; phi [rad]; strip");		
  // hh->Draw("COLZ");
  // c->SaveAs(outputDir + "phiStrip_region1_layer2_digi" + ext);

  c->Clear();		
  tree->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)","region==-1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "phiStrip_region-1_digi" + ext);

  c->Clear();		
  tree->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)","region==1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs(outputDir + "phiStrip_region1_digi" + ext);
  
  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)");
  h = (TH1D*)gDirectory->Get("h");
  title = ";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
  h->SetTitle(title);		
  h->Draw("");
  c->SaveAs(outputDir + "strip_digi" + ext);

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","region==-1");
  h = (TH1D*)gDirectory->Get("h");
  title = ";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
  h->SetTitle(title);		
  h->Draw("");
  c->SaveAs(outputDir + "strip_region-1_digi" + ext);

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","region==1");
  h = (TH1D*)gDirectory->Get("h");
  title = ";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
  h->SetTitle(title);		
  h->Draw("");
  c->SaveAs(outputDir + "strip_region1_digi" + ext);

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","layer==1");
  h = (TH1D*)gDirectory->Get("h");
  title = ";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
  h->SetTitle(title);		
  h->Draw("");
  c->SaveAs(outputDir + "strip_layer1_digi" + ext);

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","layer==2");
  h = (TH1D*)gDirectory->Get("h");
  title = ";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
  h->SetTitle(title);		
  h->Draw("");
  c->SaveAs(outputDir + "strip_layer2_digi" + ext);

  //-----------------------//
  // Bunch crossing plots  //  
  //-----------------------//

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle(";bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==1");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region-1_layer1_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==2");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region-1_layer2_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==1");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_layer1_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==2");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_layer2_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==1");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,roll1 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_roll1_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==2");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,roll2 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_roll2_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==3");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,roll3 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_roll3_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==4");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,roll4 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_roll4_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==5");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,roll5 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_roll5_digi" + ext);

  // c->Clear();		
  // tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==6");
  // h = (TH1F*)gDirectory->Get("h");
  // h->SetTitle("Bunch crossing: region1,roll6 ;bunch crossing;entries");			
  // for( unsigned uBin=0; uBin <= h->GetNbinsX(); ++uBin){
  //   h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  // }
  // h->Draw("");
  // c->SaveAs(outputDir + "bx_region1_roll6_digi" + ext);


  delete hh;
  delete h;
  delete c;
  delete tree;
  delete dir;

  return 0;
}
