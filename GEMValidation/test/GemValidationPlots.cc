
/*  Script to produce validation plots.
  Run in terminal, type:
  root -l -b
  .L GemValidation.cc
  main()
  produceSimHitValidationPlots(SimHit)
  produceSimHitValidationPlots()
  
  You will be prompted to type a production identifier, e.g. "SingleMuPt40_20121128_Pilot".
  The program will look for two files in the *working* directory:
  - SingleMuPt40_20121128_Pilot_SIM.root
  - SingleMuPt40_20121128_Pilot_DIGI.root

  EDIT: the default identifier is "gem_ana"

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
  enum selection {MUON = 0, NONMUON = 1, ALL = 2};
  
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

  // Print to .pdf file
  const bool printToPDF( true );


  {
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
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1&&abs(particleType)==13");
    TH2D *hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Muon SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf(","Title:globalxy_region-1_layer1_muon_simhit");
    else c->SaveAs( outputDir + "globalxy_region-1_layer1_muon_simhit" + ext );

    c->Clear();
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2&&abs(particleType)==13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Muon SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:globalxy_region-1_layer2_muon_simhit");
    else c->SaveAs( outputDir + "globalxy_region-1_layer2_muon_simhit" + ext);

    c->Clear();
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==1&&abs(particleType)==13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Muon SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:globalxy_region1_layer1_muon_simhit");
    else c->SaveAs( outputDir + "globalxy_region1_layer1_muon_simhit" + ext);

    c->Clear();	
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==2&&abs(particleType)==13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Muon SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:globalxy_region1_layer2_muon_simhit");
    else c->SaveAs(outputDir + "globalxy_region1_layer2_muon_simhit" + ext);
  
    //--------Non muon--------//

    c->Clear();	
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1&&abs(particleType)!=13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Non muon SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf(","Title:globalxy_region-1_layer1_nonmuon_simhit");
    else c->SaveAs( outputDir + "globalxy_region-1_layer1_nonmuon_simhit" + ext );
  
    c->Clear();
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2&&abs(particleType)!=13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Non muon SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:globalxy_region-1_layer2_nonmuon_simhit");
    else c->SaveAs( outputDir + "globalxy_region-1_layer2_nonmuon_simhit" + ext);

    c->Clear();
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==1&&abs(particleType)!=13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Non muon SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:globalxy_region1_layer1_nonmuon_simhit");
    else c->SaveAs( outputDir + "globalxy_region1_layer1_nonmuon_simhit" + ext);

    c->Clear();	
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==1&&layer==2&&abs(particleType)!=13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Non muon SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    c->SaveAs(outputDir + "globalxy_region1_layer2_nonmuon_simhit" + ext);
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:globalxy_region1_layer2_nonmuon_simhit");
    else c->SaveAs(outputDir + "globalxy_region1_layer2_nonmuon_simhit" + ext);

    //--------All--------//
  
    c->Clear();	
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==1");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("All SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf(","Title:globalxy_region-1_layer1_all_simhit");
    else c->SaveAs( outputDir + "globalxy_region-1_layer1_all_simhit" + ext );
  
    c->Clear();
    tree->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)","region==-1&&layer==2");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("All SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:globalxy_region-1_layer2_all_simhit");
    else c->SaveAs( outputDir + "globalxy_region-1_layer2_all_simhit" + ext);

    c->Clear();
    tree->Draw("globalY:globalX>>hh(100,-260,260, 100,-260,260)","region==1&&layer==1");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("All SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:globalxy_region1_layer1_all_simhit");
    else c->SaveAs( outputDir + "globalxy_region1_layer1_all_simhit" + ext);

    c->Clear();	
    tree->Draw("globalY:globalX>>hh(100,-260,260, 100,-260,260)","region==1&&layer==2");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("All SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]");   
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:globalxy_region1_layer2_all_simhit");
    else c->SaveAs(outputDir + "globalxy_region1_layer2_all_simhit" + ext);

    //--------------------//
    // ZR occupancy plots //
    //--------------------//

    //--------Muon--------//

    c->Clear();
    tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1&&abs(particleType)==13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Muon SimHit occupancy: region-1;globalZ [cm];globalR [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:globalzr_region-1_muon_simhit");
    else c->SaveAs(outputDir + "globalzr_region-1_muon_simhit" + ext);

    c->Clear();
    tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1&&abs(particleType)==13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Muon SimHit occupancy: region1;globalZ [cm];globalR [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:globalzr_region1_muon_simhit");
    else c->SaveAs(outputDir + "globalzr_region1_muon_simhit" + ext);

    //--------Non muon--------//

    c->Clear();
    tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1&&abs(particleType)!=13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Non muon SimHit occupancy: region-1;globalZ [cm];globalR [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:globalzr_region-1_all_simhit");
    else c->SaveAs(outputDir + "globalzr_region-1_nonmuon_simhit" + ext);

    c->Clear();
    tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1&&abs(particleType)!=13");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("Non muon SimHit occupancy: region1;globalZ [cm];globalR [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:globalzr_region1_all_simhit");
    else c->SaveAs(outputDir + "globalzr_region1_nonmuon_simhit" + ext);

    //--------All--------//

    c->Clear();
    tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)","region==-1");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("All SimHit occupancy: region-1;globalZ [cm];globalR [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:globalzr_region-1_all_simhit");
    else c->SaveAs(outputDir + "globalzr_region-1_all_simhit" + ext);

    c->Clear();
    tree->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)","region==1");
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle("All SimHit occupancy: region1;globalZ [cm];globalR [cm]");
    hh->Draw("COLZ");    
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:globalzr_region1_all_simhit");
    else c->SaveAs(outputDir + "globalzr_region1_all_simhit" + ext);

    //--------------------//
    // timeOfFlight plots //
    //--------------------//

    //--------Muon--------//

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1&&abs(particleType)==13");
    TH1D* h = (TH1D*)gDirectory->Get("h");
    TString title( "Muon SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns" );
    h->SetTitle( title );
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:timeOfFlight_region-1_layer1_muon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_muon_simhit" + ext);
    
    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2&&abs(particleType)==13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Muon SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );
    h->Draw("");        
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:timeOfFlight_region-1_layer2_muon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_muon_simhit" + ext);       

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1&&abs(particleType)==13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Muon SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:timeOfFlight_region1_layer1_muon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region1_layer1_muon_simhit" + ext);

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2&&abs(particleType)==13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Muon SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:timeOfFlight_region1_layer2_muon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region1_layer2_muon_simhit" + ext);
    
    //--------Non muon--------//

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1&&abs(particleType)!=13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Non muon SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );
    h->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:timeOfFlight_region-1_layer1_nonmuon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_nonmuon_simhit" + ext);

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2&&abs(particleType)!=13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Non muon SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );
    h->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:timeOfFlight_region-1_layer2_nonmuon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_nonmuon_simhit" + ext);       

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1&&abs(particleType)!=13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Non muon SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:timeOfFlight_region1_layer1_nonmuon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region1_layer1_nonmuon_simhit" + ext);

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2&&abs(particleType)!=13");
    h = (TH1D*)gDirectory->Get("h");
    title = "Non muon SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:timeOfFlight_region1_layer2_nonmuon_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region1_layer2_nonmuon_simhit" + ext);
    
    //--------All--------//

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0,30)","region==-1&&layer==1");
    h = (TH1D*)gDirectory->Get("h");
    title =  "All SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );
    h->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:timeOfFlight_region-1_layer1_all_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region-1_layer1_all_simhit" + ext);
    
    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==-1&&layer==2");
    h = (TH1D*)gDirectory->Get("h");
    title = "All SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );
    h->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:timeOfFlight_region-1_layer2_all_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region-1_layer2_all_simhit" + ext);       

    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==1");
    h = (TH1D*)gDirectory->Get("h");
    title = "All SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries/" + to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:timeOfFlight_region1_layer1_all_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region1_layer1_all_simhit" + ext);
    
    c->Clear();
    tree->Draw("timeOfFlight>>h(300,0.,30.)","region==1&&layer==2");
    h = (TH1D*)gDirectory->Get("h");
    title = "All SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries/" +  to_string(h->GetBinWidth(0)) + " ns";
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:timeOfFlight_region1_layer2_all_simhit");
    else c->SaveAs(outputDir + "timeOfFlight_region1_layer2_all_simhit" + ext);
    
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
    gPad->SetLogx();
    title = "Muon energy loss;Energy loss [eV];entries/" + to_string(h->GetBinWidth(0)) + " eV";
    muonEnergyLoss->SetTitle( title );
    muonEnergyLoss->Draw("");  
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:energyLoss_muon_simhit");
    else c->SaveAs(outputDir + "energyLoss_muon_simhit" + ext);
    
    c->Clear();
    gPad->SetLogx();
    title = "Non muon energy loss;Energy loss [eV];entries/" + to_string(h->GetBinWidth(0)) + " eV";
    nonMuonEnergyLoss->SetTitle( title );
    nonMuonEnergyLoss->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:energyLoss_nonmuon_simhit");
    else c->SaveAs(outputDir + "energyLoss_nonmuon_simhit" + ext);
    
    c->Clear();
    gPad->SetLogx();
    title = "All energy loss;Energy loss [eV];entries/" +  to_string(h->GetBinWidth(0)) + " eV";
    allEnergyLoss->SetTitle( title );
    allEnergyLoss->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:energyLoss_all_simhit");
    else c->SaveAs(outputDir + "energyLoss_all_simhit" + ext);
    
    //--------------------//
    //   momentum plots   //
    //--------------------//
  
    c->Clear();
    tree->Draw("pabs>>h(50,0.,200.)");
    h = (TH1D*)gDirectory->Get("h");
    gPad->SetLogx(0);
    title = "All SimHits absolute momentum;Momentum [eV];entries/" +  to_string(h->GetBinWidth(0)) + " eV"; // check units here
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf","Title:momentum_all_simhit");
    else c->SaveAs(outputDir + "momentum_all_simhit" + ext);

    c->Clear();
    tree->Draw("pabs>>h(50,0.,200.)","abs(particleType)==13");
    h = (TH1D*)gDirectory->Get("h");
    // gPad->SetLogx();
    title = "Muon SimHits absolute momentum;Momentum [eV];entries/" +  to_string(h->GetBinWidth(0)) + " eV"; // check units here
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf","Title:momentum_muon_simhit");
    else c->SaveAs(outputDir + "momentum_muon_simhit" + ext);
    
    c->Clear();
    tree->Draw("pabs>>h(50,0.,200.)","abs(particleType)!=13");
    h = (TH1D*)gDirectory->Get("h");
    // gPad->SetLogx();
    title = "Non muon SimHits absolute momentum;Momentum [eV];entries/" +  to_string(h->GetBinWidth(0)) + " eV"; // check units here
    h->SetTitle( title );       
    h->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf","Title:momentum_nonmuon_simhit");
    else c->SaveAs(outputDir + "momentum_nonmuon_simhit" + ext);
    
    //--------------------//
    // eta occupancy plot //
    //--------------------//

    TH1D* muonEtaOccupancy = new TH1D("muonEtaOccupancy","Muon globalEta",24,1.,25.);
    TH1D* nonMuonEtaOccupancy = new TH1D("nonMuonEtaOccupancy","Non muon globalEta",24,1.,25.);
    TH1D* allEtaOccupancy = new TH1D("allEtaOccupancy","All globalEta",24,1.,25.);
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
      if (abs(particleType)==13) muonEtaOccupancy->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
      if (abs(particleType)!=13) nonMuonEtaOccupancy->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
      allEtaOccupancy->Fill(roll + (layer==2? 6:0) + (region==1? 12:0 ) );
    }    
    gPad->SetLogx(0);
    c->Clear();  
    setEtaBinLabels(muonEtaOccupancy);
    muonEtaOccupancy->Draw("");        
    if (printToPDF) c->Print("muon_simhitValidationPlots.pdf)","Title:globalEta_muon_simhit");
    else c->SaveAs(outputDir + "globalEta_muon_simhit" + ext);
    
    c->Clear();  
    setEtaBinLabels(nonMuonEtaOccupancy);
    nonMuonEtaOccupancy->Draw("");        
    if (printToPDF) c->Print("nonmuon_simhitValidationPlots.pdf)","Title:globalEta_nonmuon_simhit");
    else c->SaveAs(outputDir + "globalEta_nonmuon_simhit" + ext);
    
    c->Clear();  
    setEtaBinLabels(allEtaOccupancy);
    allEtaOccupancy->Draw("");        
    if (printToPDF) c->Print("all_simhitValidationPlots.pdf)","Title:globalEta_all_simhit");
    else c->SaveAs(outputDir + "globalEta_all_simhit" + ext);

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
    if (printToPDF) c->Print("simhitValidationPlots.pdf","Title:particleType_simhit");
    else c->SaveAs(outputDir + "particleType_simhit" + ext);
    

    
  }

  // For studies with minbias and pileup samples we would need to be able to
  // plot distributions of number of simhits and digis per event per GEM
  // chamber and per superchamber. From these we would be able to deduce some
  // really useful probabilistic interpretations.
  // Also, it would be very useful to have distributions for the number of
  // pair-correlated simhits and pair-correlated digis per event per GEM
  // superchambers. By pair-correlated I mean here that we get matching
  // simhits or digis in both layers (chambers) of a superchamber.
  
 
  // Another interesting number to know would be how often track leaves simhits in two layers in two *different* neighboring eta partitions. The tracks are going through GEM layers at some angle, and some tracks would be traversing through neighboring eta partitions (and, BTW, it would be happening even more often when there would be more of eta partitions in GEMs). It's interesting to know the magnitude of this effect because eventually we would want to consider pair-matching of signals in  GEM layers in a superchamber, and such eta-neighbor-traversals would be the source of some inefficiency.

  const bool plotTrack( true );
  if (plotTrack)
    {
      /////////////////////////////
      // TRACK VALIDATION PLOTS //
      /////////////////////////////

      std::cout << ">>> Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
      TDirectory * dir = (TDirectory*)simHitFile->Get("gemSimHitAnalyzer");
      if (!dir){
	std::cerr << ">>> Error in GemValidationPlots::main() No such TDirectory: gemSimHitAnalyzer" << std::endl;
	exit(1);
      }

      std::cout << ">>> Reading TTree: Tracks" << std::endl;
      TTree* tree = (TTree*) dir->Get("Tracks");
      if (!tree){
	std::cerr << ">>> Error in GemValidationPlots::main() No such TTree: Tracks" << std::endl;
	exit(1);
      }
      TCanvas* c = new TCanvas("c","c",1000,600);
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Even-propagatedSimHitRhoGEMl1Even>>h(200,-2.,2.)","meanSimHitRhoGEMl1Even>0 && propagatedSimHitRhoGEMl1Even>0");
      TH1D* h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho - propagated #rho");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf(","Title:delta_meanSimHitRhoGEMl1Even_propagatedSimHitRhoGEMl1Even");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Even_propagatedSimHitRhoGEMl1Even_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl2Even-propagatedSimHitRhoGEMl2Even>>h(200,-2.,2.)","meanSimHitRhoGEMl2Even>0 && propagatedSimHitRhoGEMl2Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl2 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho - propagated #rho");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Even_propagatedSimHitRhoGEMl2Even");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl2Even_propagatedSimHitRhoGEMl2Even_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Even-propagatedSimHitEtaGEMl1Even>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Even>0 && propagatedSimHitRhoGEMl1Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta - propagated #eta");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even_propagatedSimHitEtaGEMl1Even");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Even_propagatedSimHitEtaGEMl1Even_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitEtaGEMl2Even-propagatedSimHitEtaGEMl2Even>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Even>0 && propagatedSimHitRhoGEMl2Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl2 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta - propagated #eta");
      h->SetBins(200,-1.e-4,1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Even_propagatedSimHitEtaGEMl2Even");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl2Even_propagatedSimHitEtaGEMl2Even_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Even-propagatedSimHitPhiGEMl1Even>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Even>0 && propagatedSimHitRhoGEMl1Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi - propagated #phi");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even_propagatedSimHitPhiGEMl1Even");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Even_propagatedSimHitPhiGEMl1Even_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl2Even-propagatedSimHitPhiGEMl2Even>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Even>0 && propagatedSimHitRhoGEMl2Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl2 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi - propagated #phi");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Even_propagatedSimHitPhiGEMl2Even");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl2Even-propagatedSimHitPhiGEMl2Even_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Odd-propagatedSimHitRhoGEMl1Odd>>h(200,-2.,2.)","meanSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoGEMl1Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho - propagated #rho");
      //  h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Odd_propagatedSimHitRhoGEMl1Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Odd_propagatedSimHitRhoGEMl1Odd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl2Odd-propagatedSimHitRhoGEMl2Odd>>h(200,-2.,2.)","meanSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoGEMl2Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl2 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho - propagated #rho");
      // h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Odd_propagatedSimHitRhoGEMl2Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl2Odd_propagatedSimHitRhoGEMl2Odd_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Odd-propagatedSimHitEtaGEMl1Odd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoGEMl1Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta - propagated #eta");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd_propagatedSimHitEtaGEMl1Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Odd_propagatedSimHitEtaGEMl1Odd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl2Odd-propagatedSimHitEtaGEMl2Odd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoGEMl2Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl2 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta - propagated #eta");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Odd_propagatedSimHitEtaGEMl2Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl2Odd_propagatedSimHitEtaGEMl2Odd_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Odd-propagatedSimHitPhiGEMl1Odd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Odd>0 && propagatedSimHitRhoGEMl1Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi - propagated #phi");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd_propagatedSimHitPhiGEMl1Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Odd_propagatedSimHitPhiGEMl1Odd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl2Odd-propagatedSimHitPhiGEMl2Odd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Odd>0 && propagatedSimHitRhoGEMl2Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl2 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi - propagated #phi");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Odd_propagatedSimHitPhiGEMl2Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl2Odd_propagatedSimHitPhiGEMl2Odd_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Both-propagatedSimHitRhoGEMl1Both>>h(200,-2.,2.)","meanSimHitRhoGEMl1Both>0 && propagatedSimHitRhoGEMl1Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho - propagated #rho");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Both_propagatedSimHitRhoGEMl1Both");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Both_propagatedSimHitRhoGEMl1Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl2Both-propagatedSimHitRhoGEMl2Both>>h(200,-2.,2.)","meanSimHitRhoGEMl2Both>0 && propagatedSimHitRhoGEMl2Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl2 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho - propagated #rho");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Both_propagatedSimHitRhoGEMl2Both");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl2Both_propagatedSimHitRhoGEMl2Both_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Both-propagatedSimHitEtaGEMl1Both>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Both>0 && propagatedSimHitRhoGEMl1Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta - propagated #eta");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both_propagatedSimHitEtaGEMl1Both");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Both_propagatedSimHitEtaGEMl1Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl2Both-propagatedSimHitEtaGEMl2Both>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Both>0 && propagatedSimHitRhoGEMl2Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl2 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta - propagated #eta");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Both_propagatedSimHitEtaGEMl2Both");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl2Both_propagatedSimHitEtaGEMl2Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Both-propagatedSimHitPhiGEMl1Both>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Both>0 && propagatedSimHitRhoGEMl1Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi - propagated #phi");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both_propagatedSimHitPhiGEMl1Both");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Both_propagatedSimHitPhiGEMl1Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl2Both-propagatedSimHitPhiGEMl2Both>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Both>0 && propagatedSimHitRhoGEMl2Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl2 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi - propagated #phi");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Both_propagatedSimHitPhiGEMl2Both");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl2Both_propagatedSimHitPhiGEMl2Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Even-meanSimHitRhoGEMl2Even>>h(200,-2.,2.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1-GEMl2 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho GEMl2");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Even-meanSimHitRhoGEMl2Even");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Even-meanSimHitRhoGEMl2Even_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl1 - mean #eta GEMl2");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl1 - mean #phi GEMl2");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Odd-meanSimHitRhoGEMl2Odd>>h(200,-2.,2.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1-GEMl2 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho GEMl2");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Odd-meanSimHitRhoGEMl2Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Odd-meanSimHitRhoGEMl2Odd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl1 - mean #eta GEMl2");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl1 - mean #phi GEMl2");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd_simtrack" + ext);
 
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Both-meanSimHitRhoGEMl2Both>>h(200,-2.,2.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1-GEMl2 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho GEMl2");
      h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Both-meanSimHitRhoGEMl2Both");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Both-meanSimHitRhoGEMl2Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl1 - mean #eta GEMl2");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl1 - mean #phi GEMl2");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Even-meanSimHitRhoCSCEven>>h(200,-2.,2.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1-CSC Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho CSC");
      //,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Even-meanSimHitRhoCSCEven");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Even-meanSimHitRhoCSCEven_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1-CSC Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl1 - mean #eta CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1-CSC Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl1 - mean #phi CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Odd-meanSimHitRhoCSCOdd>>h(200,-2.,2.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1-CSC Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho CSC");
      //  h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Odd-meanSimHitRhoCSCOdd");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Odd-meanSimHitRhoCSCOdd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1-CSC Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl1 - mean #eta CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1-CSC Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl1 - mean #phi CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd_simtrack" + ext);
 
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl1Both-meanSimHitRhoCSCBoth>>h(200,-2.,2.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl1-CSC Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl1 - mean #rho CSC");
      //  h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl1Both-meanSimHitRhoCSCBoth");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl1Both-meanSimHitRhoCSCBoth_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl1-CSC Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl1 - mean #eta CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl1-CSC Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl1 - mean #phi CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitRhoGEMl2Even-meanSimHitRhoCSCEven>>h(200,-2.,2.)","meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl2-CSC Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl2 - mean #rho CSC");
      //  h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Even-meanSimHitRhoCSCEven");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl2Even-meanSimHitRhoCSCEven_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl2-CSC Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl2 - mean #eta CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl2-CSC Even" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl2 - mean #phi CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl2Odd-meanSimHitRhoCSCOdd>>h(200,-2.,2.)","meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl2-CSC Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl2 - mean #rho CSC");
      //  h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Odd-meanSimHitRhoCSCOdd");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl2Odd-meanSimHitRhoCSCOdd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl2-CSC Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl2 - mean #eta CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl2-CSC Odd" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl2 - mean #phi CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd_simtrack" + ext);
  
      c->Clear();
      tree->Draw("meanSimHitRhoGEMl2Both-meanSimHitRhoCSCBoth>>h(200,-2.,2.)","meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Rho SimHit GEMl2-CSC Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #rho GEMl2 - mean #rho CSC");
      //  h->SetBins(200,-5*h->GetRMS(),5*h->GetRMS());
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitRhoGEMl2Both-meanSimHitRhoCSCBoth");
      else c->SaveAs(outputDir + "delta_meanSimHitRhoGEMl2Both-meanSimHitRhoCSCBoth_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Eta SimHit GEMl2-CSC Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #eta GEMl2 - mean #eta CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth");
      else c->SaveAs(outputDir + "delta_meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth_simtrack" + ext);

      c->Clear();
      tree->Draw("meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth>>h(200,-0.005.,0.005.)","meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both" );       
      h->GetYaxis()->SetTitle("Entries");
      h->GetXaxis()->SetTitle("mean #phi GEMl2 - mean #phi CSC");
      h->SetBins(200,-5*1.e-4,5*1.e-4);
      h->Draw("");        
      if (printToPDF) c->Print("simtrackValidationPlots.pdf)","Title:delta_meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth");
      else c->SaveAs(outputDir + "delta_meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth_simtrack" + ext);
  
      delete h;
      delete c;
      delete tree;
    }

  const bool plotDigi( true );
  if (plotDigi)
    {
      ///////////////////////////
      // DIGI VALIDATION PLOTS //
      ///////////////////////////

      std::cout << ">>> Opening TFile: " << digiFileName << std::endl;  
      std::cout << ">>> Opening TDirectory: gemDigiAnalyzer" << std::endl;  
      TDirectory* dir = (TDirectory*)digiFile->Get("gemDigiAnalyzer");
      if (!dir){
	std::cerr << ">>> Error in GemValidationPlots::main() No such TDirectory: gemDigiAnalyzer" << std::endl;
	exit(1);
      }
  
      std::cout << ">>> Reading TTree: GemSimDigiTree" << std::endl;
      TTree* tree = (TTree*) dir->Get("GEMDigiTree");
      if (!tree){
	std::cerr << ">>> Error in GemValidationPlots::main() No such TTree: GemDigiTree" << std::endl;
	exit(1);
      } 

      //--------------------//
      // XY occupancy plots //
      //--------------------//
  
      ext = ".pdf";

      TCanvas* c = new TCanvas("c","c",1000,600);
      tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==1");
      TH2D *hh = (TH2D*)gDirectory->Get("hh");  
      hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf(","Title:globalxy_region-1_layer1_all_digi");
      else c->SaveAs(outputDir + "globalxy_region-1_layer1_all_digi" + ext);

      c->Clear();
      tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==-1&&layer==2");
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:globalxy_region-1_layer2_all_digi");
      else c->SaveAs(outputDir + "globalxy_region-1_layer2_all_digi" + ext);

      c->Clear();
      tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==1");
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:globalxy_region1_layer1_all_digi");
      else c->SaveAs(outputDir + "globalxy_region1_layer1_all_digi" + ext);

      c->Clear();
      tree->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)","region==1&&layer==2");
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");	
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:globalxy_region1_layer2_all_digi");
      else c->SaveAs(outputDir + "globalxy_region1_layer2_all_digi" + ext);

      //--------------------//
      // ZR occupancy plots //
      //--------------------//

      c->Clear();		
      tree->Draw("g_r:g_z>>hh(200,-573,-564,55,130,240)","region==-1");
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region-1; globalZ [cm]; globalR [cm]");	
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:globalzr_region-1_digi");
      else c->SaveAs(outputDir + "globalzr_region-1_digi" + ext);

      c->Clear();		
      tree->Draw("g_r:g_z>>hh(200,564,573,55,130,240)","region==1");
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region1; globalZ [cm]; globalR [cm]");	
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:globalzr_region1_digi");
      else c->SaveAs(outputDir + "globalzr_region1_digi" + ext);


      //--------------------//
      //   PhiStrip plots   //
      //--------------------//
      {
      TCanvas* c = new TCanvas("c","c",1500,600);
      c->Clear();		
      tree->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)","region==-1");//
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region-1; phi [rad]; strip");		
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:phiStrip_region-1_digi");
      else c->SaveAs(outputDir + "phiStrip_region-1_digi" + ext);

      c->Clear();		
      tree->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)","region==1");//
      hh = (TH2D*)gDirectory->Get("hh");
      hh->SetTitle("Digi occupancy: region1; phi [rad]; strip");		
      hh->Draw("COLZ");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:phiStrip_region1_digi");
      else c->SaveAs(outputDir + "phiStrip_region1_digi" + ext);
  
      c->Clear();		
      tree->Draw("strip>>h(384,0.5,384.5)");
      TH1D* h = (TH1D*)gDirectory->Get("h");
      TString title = "All strips;strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
      h->SetTitle(title);		
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:strip_digi");
      else c->SaveAs(outputDir + "strip_digi" + ext);

      std::cout<< "test" << std::endl;

      c->Clear();		
      tree->Draw("strip>>h(384,0.5,384.5)","region==-1");
      h = (TH1D*)gDirectory->Get("h");
      title = "Strips region-1;strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
      h->SetTitle(title);		
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:strip_region-1_digi");
      else c->SaveAs(outputDir + "strip_region-1_digi" + ext);

      c->Clear();		
      tree->Draw("strip>>h(384,0.5,384.5)","region==1");
      h = (TH1D*)gDirectory->Get("h");
      title = "Strips region1;strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
      h->SetTitle(title);		
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:strip_region1_digi");
      else c->SaveAs(outputDir + "strip_region1_digi" + ext);

      c->Clear();		
      tree->Draw("strip>>h(384,0.5,384.5)","layer==1");
      h = (TH1D*)gDirectory->Get("h");
      title = "Strips layer1;strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
      h->SetTitle(title);		
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:strip_layer1_digi");
      else c->SaveAs(outputDir + "strip_layer1_digi" + ext);

      c->Clear();		
      tree->Draw("strip>>h(384,0.5,384.5)","layer==2");
      h = (TH1D*)gDirectory->Get("h");
      title = "Strips layer2;strip;entries/" + to_string(h->GetBinWidth(1)) + " strips";
      h->SetTitle(title);		
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:strip_layer2_digi");
      else c->SaveAs(outputDir + "strip_layer2_digi" + ext);
      }

      //-----------------------//
      // Bunch crossing plots  //  
      //-----------------------//

      const bool plotThis( true );
      if (plotThis)
	{

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)");
      TH1D* h = (TH1D*)gDirectory->Get("h");
      h->SetTitle(";bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_digi");
      else c->SaveAs(outputDir + "bx_digi" + ext);
      
      std::cout << "TEST" << std::endl;

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==1");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region-1_layer1_digi");
      else c->SaveAs(outputDir + "bx_region-1_layer1_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==-1&&layer==2");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region-1_layer2_digi");
      else c->SaveAs(outputDir + "bx_region-1_layer2_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==1");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_layer1_digi");
      else c->SaveAs(outputDir + "bx_region1_layer1_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&layer==2");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_layer2_digi");
      else c->SaveAs(outputDir + "bx_region1_layer2_digi" + ext);

	}

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==1");
      TH1D* h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,roll1 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_roll1_digi");
      else c->SaveAs(outputDir + "bx_region1_roll1_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==2");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,roll2 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_roll2_digi");
      else c->SaveAs(outputDir + "bx_region1_roll2_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==3");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,roll3 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_roll3_digi");
      else c->SaveAs(outputDir + "bx_region1_roll3_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==4");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,roll4 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_roll4_digi");
      else c->SaveAs(outputDir + "bx_region1_roll4_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==5");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,roll5 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf","Title:bx_region1_roll5_digi");
      c->SaveAs(outputDir + "bx_region1_roll5_digi" + ext);

      c->Clear();		
      tree->Draw("bx>>h(5,-2.5,2.5)","region==1&&roll==6");
      h = (TH1D*)gDirectory->Get("h");
      h->SetTitle("Bunch crossing: region1,roll6 ;bunch crossing;entries");			
      for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
	h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
      }
      h->Draw("");
      if (printToPDF) c->Print("digiValidationPlots.pdf)","Title:bx_region1_roll6_digi");      
      c->SaveAs(outputDir + "bx_region1_roll6_digi" + ext);

      delete hh;
      delete h;
      delete c;
      delete tree;
      delete dir;

    }

  return 0;
}

