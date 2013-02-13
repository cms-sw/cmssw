#include "GEMValidator.h"

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
#include "TAxis.h"
#include "TArrayD.h"
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string>
#include <sstream>
 

GEMValidator::GEMValidator()
  : fileExtension_(".pdf")
  , simHitFileName_( 
// 		    "gem_sh_ana.root" 
		    "gem_sh_ana_MuGunPt40.root"
		    )
  , digiFileName_( 
// 		  "gem_digi_ana.test.root" 
		  "gem_digi_ana_V00-02-17-MuGunPt40.root"
		  )
{
  std::cout << ">>> Using input files: " << std::endl
	    << "\t" << simHitFileName_ << std::endl
	    << "\t" << digiFileName_ << std::endl;
} 


GEMValidator::~GEMValidator()
{
}

void GEMValidator::setEtaBinLabels(const TH1D* h)
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


void GEMValidator::produceSimHitValidationPlots(const Selection& key = GEMValidator::Muon)
{
  TString particleType[] = {"&&abs(particleType)==13","&&abs(particleType)!=13",""};
  TString titlePrefix[] = {"Muon","Non muon","All"};
  std::string fileSuffix[] = {"_muon","_nonmuon","_all"};

  std::cout << ">>> Opening TFile: " << simHitFileName_ << std::endl;  
  TFile *simHitFile = new TFile(simHitFileName_);
  if (!simHitFile){
    std::cerr << "Error in GEMSimSetUp::GEMValidator() - no such TFile: " << simHitFileName_ << std::endl; 
    exit(1);
  }    

  std::cout << ">>> Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
  TDirectory * dir = (TDirectory*)simHitFile->Get("gemSimHitAnalyzer"); 
  if (!dir){
    std::cerr << ">>> Error in GemValidationPlots::produceSimHitValidationPlots() No such TDirectory: gemSimHitAnalyzer" << std::endl;
    exit(1);
  }

  std::cout << ">>> Reading TTree: GEMSimHits" << std::endl;
  TTree* tree = (TTree*) dir->Get("GEMSimHits");
  if (!tree){
    std::cerr << ">>> Error in GemValidationPlots::produceSimHitValidationPlots() No such TTree: GEMSimHits" << std::endl;
    exit(1);
  }

  std::cout << ">>> Producing PDF file: " << "simhitValidationPlots" + fileSuffix[(int)key] + ".pdf" << std::endl;

  //--------------------//
  // XY occupancy plots //
  //--------------------//

  TCanvas* c = new TCanvas("c","c",600,600);
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1" + particleType[(int)key]);
  TH2D *hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(titlePrefix[(int)key] + " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(( "simhitValidationPlots" + fileSuffix[(int)key] + ".pdf(" ).c_str(),"Title:globalxy_region-1_layer1");

  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2" + particleType[(int)key]);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(titlePrefix[(int)key] + " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");      
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalxy_region-1_layer2");
  
  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==1" + particleType[(int)key]);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(titlePrefix[(int)key] + " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalxy_region1_layer1");
  
  c->Clear();
  tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==2" + particleType[(int)key]);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(titlePrefix[(int)key] + " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
  hh->Draw("COLZ");    
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalxy_region1_layer2");
  
  //--------------------//
  // ZR occupancy plots //
  //--------------------//
  
  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1" + particleType[(int)key]);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(titlePrefix[(int)key] + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalzr_region-1");
  
  c->Clear();
  tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1" + particleType[(int)key]);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(titlePrefix[(int)key] + " SimHit occupancy: region1;globalZ [cm];globalR [cm]");
  hh->Draw("COLZ");    
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalzr_region1");
  
  //--------------------//
  // timeOfFlight plots //
  //--------------------//
  
  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1" + particleType[(int)key]);
  TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region-1_layer1");
  
  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2" + particleType[(int)key]);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region-1_layer2");
  
  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1" + particleType[(int)key]);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region1_layer1");

  c->Clear();
  tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2" + particleType[(int)key]);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region1_layer2");

  //--------------------//
  //   momentum plots   //
  //--------------------//

  c->Clear();
  tree->Draw("pabs>>h(50,0.,200.)",""+(particleType[(int)key])(2,particleType[(int)key].Length()));
  h = (TH1D*)gDirectory->Get("h");
  gPad->SetLogx(0);
  h->SetTitle( titlePrefix[(int)key] + " SimHits absolute momentum;Momentum [eV];entries/" +  to_string(h->GetBinWidth(0)) + " eV" );       
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:momentum");

  //--------------------//
  //    pdg ID plots    //
  //--------------------//
  
  c->Clear();
  gPad->SetLogx();
  // tree->Draw("particleType>>h(25,0.,25.)","abs(particleType)>0");
  tree->Draw("(particleType>0)?TMath::Log10(particleType):-TMath::Log10(particleType)>>h(200,-100.,100.)","");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "SimHit PDG Id;PDG Id;entries"  );       
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:pdgid");

  //--------------------//
  // eta occupancy plot //
  //--------------------//

  int region=0;
  int layer=0;
  int roll=0;  
  int particletype=0;
  TBranch *b_region;
  TBranch *b_layer;
  TBranch *b_roll;
  TBranch *b_particleType;  
  tree->SetBranchAddress("region", &region, &b_region);
  tree->SetBranchAddress("layer", &layer, &b_layer);
  tree->SetBranchAddress("roll", &roll, &b_roll);
  tree->SetBranchAddress("particleType", &particletype, &b_particleType);
  h = new TH1D("h", titlePrefix[(int)key] + " globalEta",24,1.,25.);
  int nbytes = 0;
  int nb = 0;
  for (Long64_t jentry=0; jentry<tree->GetEntriesFast();jentry++) {
    Long64_t ientry = tree->LoadTree(jentry);
    if (ientry < 0) break;
    nb = tree->GetEntry(jentry);   
    nbytes += nb;
    switch((int)key){
    case 0:
      if (abs(particletype)==13) h->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
      break;
    case 1:
      if (abs(particletype)!=13) h->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
      break;
    case 2:
      h->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
      break;
    }
  }    
  gPad->SetLogx(0);
  c->Clear();  
  setEtaBinLabels(h);
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalEta");

  //--------------------//
  // energy loss plots  //
  //--------------------//
  
  h = new TH1D("h","",60,0.,6000.);
  Float_t energyLoss=0;
  TBranch *b_energyLoss;
  tree->SetBranchAddress("energyLoss", &energyLoss, &b_energyLoss);
  for (Long64_t jentry=0; jentry<tree->GetEntriesFast();jentry++) {
    Long64_t ientry = tree->LoadTree(jentry);
    if (ientry < 0) break;
    nb = tree->GetEntry(jentry);   
    nbytes += nb;
    if ( particletype==13 ) h->Fill( energyLoss*1.e9 );
    if ( particletype!=13 ) h->Fill( energyLoss*1.e9 );
    h->Fill( energyLoss*1.e9 );
  }
  c->Clear();
  gPad->SetLogx();
  h->SetTitle( titlePrefix[(int)key] + " energy loss;Energy loss [eV];entries/ eV" );
  h->SetMinimum(0.);
  h->Draw("");  
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:muon_energy_plot");

  // R_0 plots

//   c->Clear();
//   tree->Draw("R_0*sin(Phi_0):R_0*sin(Phi_0)>>h(40,0.,4.)","region==-1&&layer==1" + particleType[(int)key]);
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
//   h->Draw("");        
//   c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:");

//   c->Clear();
//   tree->Draw("R_0>>h(40,0.,4.)","region==-1&&layer==1" + particleType[(int)key]);
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
//   h->Draw("");        
//   c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:");

//   c->Clear();
//   tree->Draw("R_0>>h(40,0.,4.)","region==-1&&layer==1" + particleType[(int)key]);
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
//   h->Draw("");        
//   c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:");

//   c->Clear();
//   tree->Draw("R_0>>h(40,0.,4.)","region==-1&&layer==1" + particleType[(int)key]);
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
//   h->Draw("");        
//   c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:");

  
}

void GEMValidator::produceDigiValidationPlots()
{
  std::cout << ">>> Opening TFile: " << digiFileName_ << std::endl;  
  TFile *digiFile_ = new TFile(digiFileName_);
  if (!digiFile_){
    std::cerr << "Error in GEMSimSetUp::GEMValidator() - no such TFile: " << digiFileName_ << std::endl; 
    exit(1);
  }
  
  std::cout << ">>> Opening TDirectory: gemDigiAnalyzer" << std::endl;  
  TDirectory* dir = (TDirectory*)digiFile_->Get("gemDigiAnalyzer");
  if (!dir){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TDirectory: gemDigiAnalyzer" << std::endl;
    exit(1);
  }
  
  std::cout << ">>> Reading TTree: GemDigiTree" << std::endl;
  TTree* tree = (TTree*) dir->Get("GEMDigiTree");
  if (!tree){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TTree: GemDigiTree" << std::endl;
    exit(1);
  } 
  
  std::cout << ">>> Producing PDF file: " << "digiValidationPlots.pdf" << std::endl;
  
  
  //--------------------//
  // XY occupancy plots //
  //--------------------//
  
  TCanvas* c = new TCanvas("c","c",600,600);
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==1");
  TH2D* hh = (TH2D*)gDirectory->Get("hh");  
  hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf(","Title:globalxy_region-1_layer1");

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:globalxy_region-1_layer2");

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:globalxy_region1_layer1");

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1, layer2; globalX [cm]; globalY [cm]");	
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:globalxy_region1_layer2");

  //--------------------//
  // ZR occupancy plots //
  //--------------------//

  c->Clear();		
  tree->Draw("g_r:g_z>>hh(200,-573,-564,55,130,240)","region==-1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:globalzr_region-1");

  c->Clear();		
  tree->Draw("g_r:g_z>>hh(200,564,573,55,130,240)","region==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:globalzr_region1");


  //--------------------//
  //   PhiStrip plots   //
  //--------------------//

  c->Clear();		
  tree->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)","region==-1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:phiStrip_region-1");

  c->Clear();		
  tree->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)","region==1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs("digiValidationPlots.pdf","Title:phiStrip_region1");
  
  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)");
  TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips").c_str());		
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","region==-1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips").c_str());		
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_region-1");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","region==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips").c_str());		
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_region1");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips").c_str());		
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_layer1");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";strip;entries/" + to_string(h->GetBinWidth(1)) + " strips").c_str());		
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_layer2");

  //-----------------------//
  // Bunch crossing plots  //  
  //-----------------------//

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(";bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region-1_layer1");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region-1_layer2");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_layer1");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_layer2");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll1");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll2");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll3");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll4");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll5");

  c->Clear();		
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf)","Title:bx_region1_roll6");
}

void GEMValidator::produceGEMCSCPadDigiValidationPlots(const std::string treeName)
{
  std::cout << ">>> Opening TFile: " << digiFileName_ << std::endl;  
  TFile *digiFile_ = new TFile(digiFileName_);
  if (!digiFile_){
    std::cerr << "Error in GEMSimSetUp::GEMValidator() - no such TFile: " << digiFileName_ << std::endl; 
    exit(1);
  }
  
  std::cout << ">>> Opening TDirectory: gemDigiAnalyzer" << std::endl;  
  TDirectory* dir = (TDirectory*)digiFile_->Get("gemDigiAnalyzer");
  if (!dir){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TDirectory: gemDigiAnalyzer" << std::endl;
    exit(1);
  }
  
  std::cout << ">>> Reading TTree: " << treeName << std::endl;
  TTree* tree = (TTree*) dir->Get(treeName.c_str());
  if (!tree){
    std::cerr << ">>> Error in GemValidationPlots::main() No such TTree: " << treeName << std::endl;
    exit(1);
  } 
  
  unsigned pos = treeName.find("Tree");
  TString identifier( treeName.substr(0,pos) );
  TString fileName(  identifier + "ValidationPlots.pdf");
  std::cout << ">>> Producing PDF file: " << fileName << std::endl;
  

  //--------------------//
  // XY occupancy plots //
  //--------------------//
  
  TCanvas* c = new TCanvas("c","c",600,600);
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==1");
  TH2D* hh = (TH2D*)gDirectory->Get("hh");  
  hh->SetTitle(identifier + " occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs(fileName + "(","Title:globalxy_region-1_layer1");

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:globalxy_region-1_layer2");

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:globalxy_region1_layer1");

  c->Clear();
  tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==2");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region1, layer2; globalX [cm]; globalY [cm]");	
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:globalxy_region1_layer2");

  //--------------------//
  // ZR occupancy plots //
  //--------------------//

  c->Clear();		
  tree->Draw("g_r:g_z>>hh(200,-573,-564,55,130,240)","region==-1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region-1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:globalzr_region-1");

  c->Clear();		
  tree->Draw("g_r:g_z>>hh(200,564,573,55,130,240)","region==1");
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:globalzr_region1");


  //--------------------//
  //   PhiPad plots   //
  //--------------------//

  c->Clear();		
  tree->Draw("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,25,0,25)","region==-1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region-1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:phiPad_region-1");

  c->Clear();		
  tree->Draw("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,25,0,25)","region==1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:phiPad_region1");
  
  c->Clear();		
  tree->Draw("pad>>h(25,0.5,25.5)");
  TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->Draw("");
  c->SaveAs(fileName,"Title:pad");

  c->Clear();		
  tree->Draw("pad>>h(25,0.5,25.5)","region==-1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_region-1");

  c->Clear();		
  tree->Draw("pad>>h(25,0.5,25.5)","region==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_region1");

  c->Clear();		
  tree->Draw("pad>>h(25,0.5,25.5)","layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_layer1");

  c->Clear();		
  tree->Draw("pad>>h(25,0.5,25.5)","layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle((";pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_layer2");

  //-----------------------//
  // Bunch crossing plots  //  
  //-----------------------//

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(";bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region-1_layer1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region-1_layer2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_layer1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_layer2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll3");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll4");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll5");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->Draw("");
  c->SaveAs(fileName + ")","Title:bx_region1_roll6");

}


void GEMValidator::produceTrackValidationPlots()
{
  std::cout << ">>> Opening TFile: " << simHitFileName_ << std::endl;  
  TFile *simHitFile = new TFile(simHitFileName_);
  if (!simHitFile){
    std::cerr << "Error in GEMValidator::produceTrackPlots() No such TFile: " << simHitFileName_ << std::endl; 
    exit(1);
  }    

  std::cout << ">>> Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
  TDirectory * dir = (TDirectory*)simHitFile->Get("gemSimHitAnalyzer"); 
  if (!dir){
    std::cerr << ">>> Error in GEMValidator::produceTrackPlots() No such TDirectory: gemSimHitAnalyzer" << std::endl;
    exit(1);
  }

  std::cout << ">>> Reading TTree: Tracks" << std::endl;
  TTree* tree = (TTree*) dir->Get("Tracks");
  if (!tree){
    std::cerr << ">>> Error in GEMValidator::produceTrackPlots() No such TTree: Tracks" << std::endl;
    exit(1);
  }

  std::cout << ">>> Producing PDF file: " << "trackValidationPlots.pdf" << std::endl;

  /////////////////////////////
  // simhit - simtrack GEML1 //
  /////////////////////////////

  TCanvas* c = new TCanvas("c","c",500,600);
  c->Clear();
  tree->Draw("meanSimHitRhoGEMl1Even-propagatedSimHitRhoGEMl1Even",
	     "meanSimHitRhoGEMl1Even>0 && propagatedSimHitRhoGEMl1Even>0");
  TH1D* h = (TH1D*)gPad->FindObject("htemp");
  h->SetBins(200,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS());
//   TH1F* hclonel1_CSC_Odd_positive_pT5= h->Clone();
//   hist2 = new TH1F("hist","",100,(htemp->GetMean()-10*htemp->GetRMS()),(htemp->GetMean()+10*htemp->GetRMS()));
//   hist2->SetMinimum(1); 
//   hist2->SetMaximum(htemp->GetEntries()/3.);
//   hist2->SetTitle("Odd Chambers, Charge = +1, p_{T} = 5 GeV;Mean #phi_{CSC}^{SimHit} - Mean #phi_{GEM}^{SimHit} [radians];Entries");
//   hist2->GetYaxis()->SetTitleOffset(1.7);
//   hist2->Draw("");
//   hl1_CSC_Odd_positive_pT5->Draw("same");


//   TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1 Even; mean #rho - propagated #rho [cm]; Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf(","Title:delta_meanSimHitRhoGEMl1Even_propagatedSimHitRhoGEMl1Even");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Even-propagatedSimHitEtaGEMl1Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && propagatedSimHitRhoGEMl1Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1 Even; mean #eta - propagated #eta; Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even_propagatedSimHitEtaGEMl1Even");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Even-propagatedSimHitPhiGEMl1Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && propagatedSimHitRhoGEMl1Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1 Even; mean #phi - propagated #phi [rad]; Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even_propagatedSimHitPhiGEMl1Even");

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl1Odd-propagatedSimHitRhoGEMl1Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoGEMl1Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1 Odd;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Odd_propagatedSimHitRhoGEMl1Odd");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Odd-propagatedSimHitEtaGEMl1Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoGEMl1Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1 Odd; mean #eta - propagated #eta; Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd_propagatedSimHitEtaGEMl1Odd");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Odd-propagatedSimHitPhiGEMl1Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoGEMl1Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1 Odd;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd_propagatedSimHitPhiGEMl1Odd");

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl1Both-propagatedSimHitRhoGEMl1Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && propagatedSimHitRhoGEMl1Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1 Both;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Both_propagatedSimHitRhoGEMl1Both");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Both-propagatedSimHitEtaGEMl1Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && propagatedSimHitRhoGEMl1Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1 Both;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both_propagatedSimHitEtaGEMl1Both");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Both-propagatedSimHitPhiGEMl1Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && propagatedSimHitRhoGEMl1Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1 Both;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both_propagatedSimHitPhiGEMl1Both");


  /////////////////////////////
  // simhit - simtrack GEML2 //
  /////////////////////////////

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl2Even-propagatedSimHitRhoGEMl2Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Even>0 && propagatedSimHitRhoGEMl2Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2 Even;Entries;mean #rho - propagated #rho [cm]" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Even_propagatedSimHitRhoGEMl2Even");
  
  c->Clear();
  tree->Draw("meanSimHitEtaGEMl2Even-propagatedSimHitEtaGEMl2Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Even>0 && propagatedSimHitRhoGEMl2Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2 Even;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Even_propagatedSimHitEtaGEMl2Even");
  
  c->Clear();
  tree->Draw("meanSimHitPhiGEMl2Even-propagatedSimHitPhiGEMl2Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Even>0 && propagatedSimHitRhoGEMl2Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2 Even;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Even-propagatedSimHitPhiGEMl2Even");

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl2Odd-propagatedSimHitRhoGEMl2Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoGEMl2Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2 Odd;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Odd_propagatedSimHitRhoGEMl2Odd");
  
  c->Clear();
  tree->Draw("meanSimHitEtaGEMl2Odd-propagatedSimHitEtaGEMl2Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoGEMl2Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2 Odd;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Odd_propagatedSimHitEtaGEMl2Odd");
  
  c->Clear();
  tree->Draw("meanSimHitPhiGEMl2Odd-propagatedSimHitPhiGEMl2Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoGEMl2Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2 Odd;mean #phi - propagated #phi [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Odd_propagatedSimHitPhiGEMl2Odd");
  
  c->Clear();
  tree->Draw("meanSimHitRhoGEMl2Both-propagatedSimHitRhoGEMl2Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Both>0 && propagatedSimHitRhoGEMl2Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2 Both;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Both_propagatedSimHitRhoGEMl2Both");
  
  c->Clear();
  tree->Draw("meanSimHitEtaGEMl2Both-propagatedSimHitEtaGEMl2Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Both>0 && propagatedSimHitRhoGEMl2Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2 Both;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Both_propagatedSimHitEtaGEMl2Both");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl2Both-propagatedSimHitPhiGEMl2Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Both>0 && propagatedSimHitRhoGEMl2Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2 Both;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Both_propagatedSimHitPhiGEMl2Both");


  ///////////////////////////
  // simhit - simtrack CSC //
  ///////////////////////////

  c->Clear();
  tree->Draw("meanSimHitRhoCSCEven-propagatedSimHitRhoCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCEven>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit CSC Even;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoCSCEven_propagatedSimHitRhoCSCEven");
  
  c->Clear();
  tree->Draw("meanSimHitEtaCSCEven-propagatedSimHitEtaCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCEven>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit CSC Even;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaCSCEven_propagatedSimHitEtaCSCEven");
  
  c->Clear();
  tree->Draw("meanSimHitPhiCSCEven-propagatedSimHitPhiCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCEven>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit CSC Even;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiCSCEven-propagatedSimHitPhiCSCEven");

  c->Clear();
  tree->Draw("meanSimHitRhoCSCOdd-propagatedSimHitRhoCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCOdd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit CSC Odd;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoCSCOdd_propagatedSimHitRhoCSCOdd");
  
  c->Clear();
  tree->Draw("meanSimHitEtaCSCOdd-propagatedSimHitEtaCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCOdd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit CSC Odd;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaCSCOdd_propagatedSimHitEtaCSCOdd");
  
  c->Clear();
  tree->Draw("meanSimHitPhiCSCOdd-propagatedSimHitPhiCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCOdd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit CSC Odd;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiCSCOdd_propagatedSimHitPhiCSCOdd");
  
  c->Clear();
  tree->Draw("meanSimHitRhoCSCBoth-propagatedSimHitRhoCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCBoth>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit CSC Both;mean #rho - propagated #rho [cm];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoCSCBoth_propagatedSimHitRhoCSCBoth");
  
  c->Clear();
  tree->Draw("meanSimHitEtaCSCBoth-propagatedSimHitEtaCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCBoth>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit CSC Both;mean #eta - propagated #eta;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaCSCBoth_propagatedSimHitEtaCSCBoth");

  c->Clear();
  tree->Draw("meanSimHitPhiCSCBoth-propagatedSimHitPhiCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoCSCBoth>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit CSC Both;mean #phi - propagated #phi [rad];Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiCSCBoth_propagatedSimHitPhiCSCBoth");


  //////////////////////////
  // SimHit GEML1 - GEML2 //
  //////////////////////////

  //   c->Clear();
  //   tree->Draw("meanSimHitRhoGEMl1Even-meanSimHitRhoGEMl2Even>>h(200,-2.,2.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
  //   h = (TH1D*)gDirectory->Get("h");
  //   h->SetTitle( "Delta Rho SimHit GEMl1-GEMl2 Even" );       
  //   h->GetYaxis()->SetTitle("Entries");
  //   h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho GEMl2");
  //   h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
  //   h->Draw("");        
  //   c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Even-meanSimHitRhoGEMl2Even");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Even;mean #eta GEMl1 - mean #eta GEMl2;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Even;mean #phi GEMl1 - mean #phi GEMl2;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even");
  
  //   c->Clear();
  //   tree->Draw("meanSimHitRhoGEMl1Odd-meanSimHitRhoGEMl2Odd>>h(200,-2.,2.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
  //   h = (TH1D*)gDirectory->Get("h");
  //   h->SetTitle( "Delta Rho SimHit GEMl1-GEMl2 Odd" );       
  //   h->GetYaxis()->SetTitle("Entries");
  //   h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho GEMl2");
  //   h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
  //   h->Draw("");        
  //   c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Odd-meanSimHitRhoGEMl2Odd");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Odd;mean #eta GEMl1 - mean #eta GEMl2;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Odd;mean #phi GEMl1 - mean #phi GEMl2;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd");
 
//   c->Clear();
//   tree->Draw("meanSimHitRhoGEMl1Both-meanSimHitRhoGEMl2Both>>h(200,-2.,2.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Rho SimHit GEMl1-GEMl2 Both" );       
//   h->GetYaxis()->SetTitle("Entries");
//   h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho GEMl2");
//   h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
//   h->Draw("");        
//   c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Both-meanSimHitRhoGEMl2Both");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Both;mean #eta GEMl1 - mean #eta GEMl2;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Both;mean #phi GEMl1 - mean #phi GEMl2;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both");


  ////////////////////////
  // simhit GEML1 - CSC //
  ////////////////////////

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl1Even-meanSimHitRhoCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1-CSC Even;mean #rho GEMl1 - mean #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Even-meanSimHitRhoCSCEven");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-CSC Even;mean #eta GEMl1 - mean #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-CSC Even;mean #phi GEMl1 - mean #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven");
  
  c->Clear();
  tree->Draw("meanSimHitRhoGEMl1Odd-meanSimHitRhoCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1-CSC Odd;mean #rho GEMl1 - mean #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Odd-meanSimHitRhoCSCOdd");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-CSC Odd;mean #eta GEMl1 - mean #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-CSC Odd;mean #phi GEMl1 - mean #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd");
 
  c->Clear();
  tree->Draw("meanSimHitRhoGEMl1Both-meanSimHitRhoCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1-CSC Both;mean #rho GEMl1 - mean #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Both-meanSimHitRhoCSCBoth");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-CSC Both;mean #eta GEMl1 - mean #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-CSC Both;mean #phi GEMl1 - mean #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth");


  ////////////////////////
  // simhit GEML2 - CSC //
  ////////////////////////

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl2Even-meanSimHitRhoCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2-CSC Even;mean #rho GEMl2 - mean #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Even-meanSimHitRhoCSCEven");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2-CSC Even;mean #eta GEMl2 - mean #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2-CSC Even;mean #phi GEMl2 - mean #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven");

  c->Clear();
  tree->Draw("meanSimHitRhoGEMl2Odd-meanSimHitRhoCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2-CSC Odd;mean #rho GEMl2 - mean #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Odd-meanSimHitRhoCSCOdd");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2-CSC Odd;mean #eta GEMl2 - mean #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2-CSC Odd;mean #phi GEMl2 - mean #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd");
  
  c->Clear();
  tree->Draw("meanSimHitRhoGEMl2Both-meanSimHitRhoCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2-CSC Both;mean #rho GEMl2 - mean #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Both-meanSimHitRhoCSCBoth");

  c->Clear();
  tree->Draw("meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2-CSC Both;mean #eta GEMl2 - mean #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth");

  c->Clear();
  tree->Draw("meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;mean #phi GEMl2 - mean #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth");


  ////////////////////////
  // simtrack GEML1-CSC //  
  ////////////////////////

  c->Clear();
  tree->Draw("propagatedSimHitRhoGEMl1Even-propagatedSimHitRhoCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Even>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1-CSC Even;propagated #rho GEMl1 - propagated #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitRhoGEMl1Even-propagatedSimHitRhoCSCEven");

  c->Clear();
  tree->Draw("propagatedSimHitEtaGEMl1Even-propagatedSimHitEtaCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Even>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-CSC Even;propagated #eta GEMl1 - propagated #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitEtaGEMl1Even-propagatedSimHitEtaCSCEven");

  c->Clear();
  tree->Draw("propagatedSimHitPhiGEMl1Even-propagatedSimHitPhiCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Even>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-CSC Even;propagated #phi GEMl1 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitPhiGEMl1Even-propagatedSimHitPhiCSCEven");

  c->Clear();
  tree->Draw("propagatedSimHitRhoGEMl1Odd-propagatedSimHitRhoCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1-CSC Odd;propagated #rho GEMl1 - propagated #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitRhoGEMl1Odd-propagatedSimHitRhoCSCOdd");

  c->Clear();
  tree->Draw("propagatedSimHitEtaGEMl1Odd-propagatedSimHitEtaCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-CSC Odd;propagated #eta GEMl1 - propagated #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitEtaGEMl1Odd-propagatedSimHitEtaCSCOdd");

  c->Clear();
  tree->Draw("propagatedSimHitPhiGEMl1Odd-propagatedSimHitPhiCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-CSC Odd;propagated #phi GEMl1 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitPhiGEMl1Odd-propagatedSimHitPhiCSCOdd");
  
  c->Clear();
  tree->Draw("propagatedSimHitRhoGEMl1Both-propagatedSimHitRhoCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Both>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl1-CSC Both;propagated #rho GEMl1 - propagated #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitRhoGEMl1Both-propagatedSimHitRhoCSCBoth");

  c->Clear();
  tree->Draw("propagatedSimHitEtaGEMl1Both-propagatedSimHitEtaCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Both>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl1-CSC Both;propagated #eta GEMl1 - propagated #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitEtaGEMl1Both-propagatedSimHitEtaCSCBoth");

  c->Clear();
  tree->Draw("propagatedSimHitPhiGEMl1Both-propagatedSimHitPhiCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl1Both>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl1-CSC Both;propagated #phi GEMl1 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitPhiGEMl1Both-propagatedSimHitPhiCSCBoth");

  
  ////////////////////////
  // simtrack GEML2-CSC //  
  ////////////////////////

  c->Clear();
  tree->Draw("propagatedSimHitRhoGEMl2Even-propagatedSimHitRhoCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Even>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2-CSC Even;propagated #rho GEMl2 - propagated #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitRhoGEMl2Even-propagatedSimHitRhoCSCEven");

  c->Clear();
  tree->Draw("propagatedSimHitEtaGEMl2Even-propagatedSimHitEtaCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Even>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2-CSC Even;propagated #eta GEMl2 - propagated #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitEtaGEMl2Even-propagatedSimHitEtaCSCEven");

  c->Clear();
  tree->Draw("propagatedSimHitPhiGEMl2Even-propagatedSimHitPhiCSCEven>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Even>0 && propagatedSimHitRhoCSCEven>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2-CSC Even;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitPhiGEMl2Even-propagatedSimHitPhiCSCEven");

  c->Clear();
  tree->Draw("propagatedSimHitRhoGEMl2Odd-propagatedSimHitRhoCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2-CSC Odd;propagated #rho GEMl2 - propagated #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitRhoGEMl2Odd-propagatedSimHitRhoCSCOdd");

  c->Clear();
  tree->Draw("propagatedSimHitEtaGEMl2Odd-propagatedSimHitEtaCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2-CSC Odd;propagated #eta GEMl2 - propagated #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitEtaGEMl2Odd-propagatedSimHitEtaCSCOdd");

  c->Clear();
  tree->Draw("propagatedSimHitPhiGEMl2Odd-propagatedSimHitPhiCSCOdd>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoCSCOdd>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2-CSC Odd;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitPhiGEMl2Odd-propagatedSimHitPhiCSCOdd");
  
  c->Clear();
  tree->Draw("propagatedSimHitRhoGEMl2Both-propagatedSimHitRhoCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())","propagatedSimHitRhoGEMl2Both>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Rho SimHit GEMl2-CSC Both;propagated #rho GEMl2 - propagated #rho CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitRhoGEMl2Both-propagatedSimHitRhoCSCBoth");

  c->Clear();
  tree->Draw("propagatedSimHitEtaGEMl2Both-propagatedSimHitEtaCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl2Both>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Eta SimHit GEMl2-CSC Both;propagated #eta GEMl2 - propagated #eta CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitEtaGEMl2Both-propagatedSimHitEtaCSCBoth");

  c->Clear();
  tree->Draw("propagatedSimHitPhiGEMl2Both-propagatedSimHitPhiCSCBoth>>h(100,h->GetMean()-5*h->GetRMS(),h->GetMean()+5*h->GetRMS())",
	     "propagatedSimHitRhoGEMl2Both>0 && propagatedSimHitRhoCSCBoth>0");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:delta_propagatedSimHitPhiGEMl2Both-propagatedSimHitPhiCSCBoth");

  // efficiency for a track to leave simhits in eta and phi ODD
//   c->Clear();		
//   tree->Draw("hasGEMl1:eta>>g(100,1.6,2.5,4,0.,4.)","region==1&&hasGEMl1==1");//,"&&roll==6"
//   TH2D* g = (TH2D*)gDirectory->Get("g");
//   g->Draw("");
//   c->SaveAs("trackValidationPlots.pdf","Title:trackefficiency");
  // hasGEMl1 : track with simhits in l1 in GEM superchamber 0,1,2,3 none, odd, even, both
  // hasGEMl2 : track with simhits in l2 in GEM superchamber 
  // hasCSC   : track with simhits in CSC

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.4)","hasGEMl1==1&&hasGEMl2==1");
  h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:track1");
  
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.4)","hasGEMl1==2&&hasGEMl2==2");
  h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:track2");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.4)","hasGEMl1==0&&hasGEMl2==0");
  h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:track1");
  
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.4)","hasGEMl1==3&&hasGEMl2==3");
  h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf)","Title:track2");

  
//   c->Clear();
//   tree->Draw("eta>>h(100,1.6,2.4)");
//   h = (TH1D*)gDirectory->Get("h");
// //   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;propagated #phi GEMl2 - propagated #phi CSC;Entries" );       
//   h->Draw("");        
//   c->SaveAs("trackValidationPlots.pdf","Title:track2");


  return;
}


template<typename T> const std::string GEMValidator::to_string( T const& value )
{
  std::stringstream sstr;
  sstr << value;
  return sstr.str();
}
