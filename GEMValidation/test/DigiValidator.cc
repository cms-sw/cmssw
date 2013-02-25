#include "DigiValidator.h"

DigiValidator::DigiValidator()
{
  
}

DigiValidator::~DigiValidator()
{
  
}

void DigiValidator::makeValidationPlots()
{
  std::cout << ">>> Opening TFile: " << getInFileName() << std::endl;  
  TFile *digiFile_ = new TFile(getInFileName().c_str());
  if (!digiFile_){
    std::cerr << "Error in GEMSimSetUp::GEMValidator() - no such TFile: " << getInFileName() << std::endl; 
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
  
  
  ////////////////////////
  // XY occupancy plots //
  ////////////////////////
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

  ////////////////////////
  // ZR occupancy plots //
  ////////////////////////
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


  ////////////////////
  // PhiStrip plots //
  ////////////////////
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
  h->SetTitle("Digi occupancy per strip number;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","region==-1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, region-1;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_region-1");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","region==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, region1;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_region1");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, layer1;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_layer1");

  c->Clear();		
  tree->Draw("strip>>h(384,0.5,384.5)","layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, layer2;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:strip_layer2");

  //////////////////////////
  // Bunch crossing plots //  
  //////////////////////////
  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(";bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==-1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region-1_layer1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==-1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region-1_layer2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_layer1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_layer2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll3");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll4");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf","Title:bx_region1_roll5");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.5,5.5)","region==1&&roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs("digiValidationPlots.pdf)","Title:bx_region1_roll6");
  
}

void DigiValidator::makeGEMCSCPadDigiValidationPlots(const std::string treeName)
{
  std::cout << ">>> Opening TFile: " << getInFileName() << std::endl;  
  TFile *digiFile_ = new TFile((getInFileName()).c_str());
  if (!digiFile_){
    std::cerr << "Error in GEMValidator::makeGEMCSCPadDigiValidationPlots - no such TFile: " << getInFileName() << std::endl; 
    return;
  }
  
  std::cout << ">>> Opening TDirectory: gemDigiAnalyzer" << std::endl;  
  TDirectory* dir = (TDirectory*)digiFile_->Get("gemDigiAnalyzer");
  if (!dir){
    std::cerr << ">>> Error in GEMValidator::makeGEMCSCPadDigiValidationPlots No such TDirectory: gemDigiAnalyzer" << std::endl;
    return;
  }
  
  std::cout << ">>> Reading TTree: " << treeName << std::endl;
  TTree* tree = (TTree*) dir->Get(treeName.c_str());
  if (!tree){
    std::cerr << ">>> Error in GEMValidator::makeGEMCSCPadDigiValidationPlots No such TTree: " << treeName << std::endl;
    return;
  } 
  
  unsigned pos = treeName.find("Tree");
  TString identifier( treeName.substr(0,pos) );
  TString fileName(  identifier + "ValidationPlots.pdf");

  std::cout << ">>> Producing PDF file: " << fileName << std::endl;

  ////////////////////////
  // XY occupancy plots //
  ////////////////////////
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

  ////////////////////////
  // ZR occupancy plots //
  ////////////////////////
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

  ///////////////////
  // Phi pad plots //
  ///////////////////
  c->Clear();		
  tree->Draw("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,24,0,24)","region==-1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region-1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:phiPad_region-1");

  c->Clear();		
  tree->Draw("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,24,0,24)","region==1");//
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle(identifier + " occupancy: region1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs(fileName,"Title:phiPad_region1");
  
  c->Clear();		
  tree->Draw("pad>>h(24,0.5,24.5)");
  TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Digi occupancy per pad;pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs(fileName,"Title:pad");

  c->Clear();		
  tree->Draw("pad>>h(24,0.5,24.5)","region==-1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Digi occupancy per pad - region-1;pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_region-1");

  c->Clear();		
  tree->Draw("pad>>h(24,0.5,24.5)","region==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Digi occupancy per pad - region1;pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_region1");

  c->Clear();		
  tree->Draw("pad>>h(24,0.5,24.5)","layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Digi occupancy per pad - layer1;pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_layer1");

  c->Clear();		
  tree->Draw("pad>>h(24,0.5,24.5)","layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Digi occupancy per pad - layer2;pad;entries/" + to_string(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs(fileName,"Title:pad_layer2");

  //////////////////////////
  // Bunch crossing plots //  
  //////////////////////////
  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(";bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==-1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer1;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region-1_layer1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==-1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1,layer2; bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region-1_layer2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&layer==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_layer1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&layer==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,layer2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_layer2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll1");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll2");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll3");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll4");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName,"Title:bx_region1_roll5");

  c->Clear();		
  gPad->SetLogy();
  tree->Draw("bx>>h(11,-5.,5.)","region==1&&roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1,roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, to_string( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs(fileName + ")","Title:bx_region1_roll6");
}

void DigiValidator::makeTrackValidationPlots()
{
  std::cout << ">>> Opening TFile: " << getInFileName() << std::endl;  
  TFile *digiFile = new TFile((getInFileName()).c_str());
  if (!digiFile){  
    std::cerr << "Error in DigiValidator::makeTrackValidationPlots() - no such TFile: " << getInFileName() << std::endl; 
    return;
  }    
  
  std::cout << ">>> Opening TDirectory: gemDigiAnalyzer" << std::endl;  
  TDirectory * dir = (TDirectory*)digiFile->Get("gemDigiAnalyzer"); 
  if (!dir){
    std::cerr << ">>> Error in DigiValidator::makeTrackValidationPlots() - No such TDirectory: gemDigiAnalyzer" << std::endl;
    return;
  }

  std::cout << ">>> Reading TTree: TrackTree" << std::endl;
  TTree* tree = (TTree*) dir->Get("TrackTree");
  if (!tree){
    std::cerr << ">>> Error in DigiValidator::makeTrackValidationPlots() - No such TTree: TrackTree" << std::endl;
    return;
  }

  std::cout << ">>> Producing PDF file: " << "trackValidationPlots.pdf" << std::endl;

  TCanvas* c = new TCanvas("c","c",600,600);
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==1");//""
  TH1D* h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_dg_layer==1");
  TH1D* g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf(","Title:eff_eta_tracks_digi_gem_layer1");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==1");//""
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_dg_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_digi_gem_layer1");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==2");//""
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_dg_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_digi_gem_layer2");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==2");//""
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_dg_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_digi_gem_layer2");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","has_gem_dg2==0||has_gem_dg2==1||has_gem_dg2==2");//""
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 or GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_digi_gem_layer12");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","has_gem_dg2==0||has_gem_dg2==1||has_gem_dg2==2");//""
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 or GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_digi_gem_layer12");

  // plots for matching efficiency of simtrack to digi with matched simhits
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_dg_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_sim_digi_gem_layer1");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_dg_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_sim_digi_gem_layer1");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_dg_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl2 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_sim_digi_gem_layer2");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_dg==0||has_gem_dg==1||has_gem_dg==2)&&gem_dg_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_dg_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_sim_digi_gem_layer2");

  // efficiency to match a simtrack to pad digi 
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_dg_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_dg_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_pad_gem_layer1");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_dg_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_dg_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_pad_gem_layer2");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_dg_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_dg_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_pad_gem_layer2");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_dg_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_dg_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_pad_gem_layer2");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 or GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_pad_gem_layer12");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 or GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_pad_gem_layer12");

  // pad digi plots with simhit selection
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_pad_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_pad_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_sim_pad_gem_layer1");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_pad_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_pad_layer==1&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_sim_pad_gem_layer1");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_pad_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_pad_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_sim_pad_gem_layer2");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","(has_gem_pad==0||has_gem_pad==1||has_gem_pad==2)&&gem_pad_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312","gem_pad_layer==2&&(has_gem_sh==0||has_gem_sh==1||has_gem_sh==2)&&gem_sh_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_sim_pad_gem_layer2");

  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","has_gem_copad==0||has_gem_copad==1||has_gem_copad==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Co-Pad matching efficiency;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_copad_gem");

  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312","has_gem_copad==0||has_gem_copad==1||has_gem_copad==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Co-Pad matching efficiency;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf)","Title:eff_phi_tracks_copad_gem");


}

void DigiValidator::makeValidationReport()
{
  ofstream file;
  file.open((getOutFileName()).c_str());

  file << "\\documentclass[11pt]{report}" << std::endl
       << "\\usepackage{a4wide}" << std::endl
       << "\\usepackage[affil-it]{authblk}" << std::endl
       << "\\usepackage{amsmath}" << std::endl
       << "\\usepackage{amsfonts}" << std::endl
       << "\\usepackage{amssymb}" << std::endl
       << "\\usepackage{makeidx}" << std::endl
       << "\\usepackage{graphicx}" << std::endl
       << "\\usepackage{verbatim}" << std::endl
       << "\\usepackage[T1]{fontenc}" << std::endl
       << "\\usepackage[utf8]{inputenc}" << std::endl
       << "\\usepackage{hyperref}" << std::endl
       << "\\usepackage[section]{placeins}" << std::endl;
    
  file << "\\title{\\LARGE\\textbf{CMS GEM Collaboration} \\\\[0.2cm] \\Large (GEMs for CMS) \\\\[0.2cm] \\LARGE\\textbf{Production Report}}" << std::endl;

  file << "\\author[1]{Yasser~Assran}" << std::endl
       << "\\author[2]{Othmane~Bouhali}" << std::endl
       << "\\author[3]{Sven~Dildick}" << std::endl
       << "\\author[4]{Will~Flanagan}" << std::endl
       << "\\author[4]{Teruki~Kamon}" << std::endl
       << "\\author[4]{Vadim~Khotilovich}" << std::endl
       << "\\author[4]{Roy~Montalvo}" << std::endl
       << "\\author[4]{Alexei~Safonov}" << std::endl;
    
  file << "\\affil[1]{ASRT-ENHEP (Egypt)}" << std::endl
       << "\\affil[2]{ITS Research Computing, Texas A\\&M University at Qatar (Qatar)}" << std::endl
       << "\\affil[3]{Department of Physics and Astronomy, Ghent University (Belgium)}" << std::endl
       << "\\affil[4]{Department of Experimental High Energy Physics, Texas A\\&M University (USA)}" << std::endl;

  file << "\\date{February 5, 2013 \\\\[1cm] Contact: \\href{mailto:gem-sim-validation@cern.ch}{gem-sim-validation@cern.ch}}" << std::endl;
  
  file << "\\renewcommand\\Authands{ and }" << std::endl 
       << "\\renewcommand{\\thesection}{\\arabic{section}}" << std::endl;

  file << "\\begin{document}" << std::endl;
  
  file << "\\maketitle" << std::endl;

  file << "\\section{Production information}" << std::endl;

  // replace all occurences of "_" to "\\_" in dataset path - otherwise problem in LaTeX
  std::string dsp = getDataSetPath();
  size_t pos = 0;
  std::string oldStr = "_";
  std::string newStr = "\\_";
  while((pos = dsp.find(oldStr, pos)) != std::string::npos){
    dsp.replace(pos, oldStr.length(), newStr);
    pos += newStr.length();
  }

  file << "\\begin{description}" << std::endl 
       << "\\item[Title:] " << getTitle() << std::endl
       << "\\item[Priority:] " << getPriority() << std::endl
       << "\\item[Date of request:] " << getDateOfRequest() << std::endl
       << "\\item[Description:] " << getDescription() << std::endl
       << "\\item[Link to Twiki:] \\href{" << getLinkToTwiki() << "}{" << getLinkToTwiki() << "}" << std::endl
       << "\\item[Production start date:] " << getProductionStartDate() << std::endl
       << "\\item[Responsible:] " << getResponsible() << std::endl
       << "\\item[Production end date:] " << getProductionEndDate() << std::endl
       << "\\item[Data set path:] {\\scriptsize \\texttt{" << dsp << "}}"  << std::endl
       << "\\item[Time to complete:] " << getTimeToComplete() << std::endl
       << "\\item[Crab configuration:] " << getCrabConfiguration() << std::endl;
  if (isObsolete()){
    file << "\\item[Obsolete:] " << "Yes"  << std::endl
	 <<  "\\item[Date of obsoletion:] " << getDateOfObsoletion() << std::endl
	 << "\\item[Reason for obsoletion:] " << getReasonForObsoletion() << std::endl
	 << "\\item[Deleted: ] " << (isDeleted() ? "Yes" : "No" ) << std::endl;
  }
  else{
    file << "\\item[Obsolete:] " << "No"  << std::endl
	 <<  "\\item[Date of obsoletion:] N/A" << std::endl
	 << "\\item[Reason for obsoletion:] N/A" << std::endl
	 << "\\item[Deleted: ] N/A" << std::endl;    
  }
  
  file << "\\end{description}"  << std::endl;
  file << std::endl;

  file << "\\newpage" << std::endl;

  // Digi validation plots

  file << "\\section{Digi validation plots}" << std::endl;
  
  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{digiValidationPlots.pdf} " << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{digiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{digiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl
       << std::endl;
  
  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=7]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=8]{digiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=10]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{digiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=12]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=13]{digiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.5\\textwidth,page=15]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.5\\textwidth,page=16]{digiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.5\\textwidth,page=17]{digiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.5\\textwidth,page=18]{digiValidationPlots.pdf}" << std::endl
//        << "\\\\" << std::endl
//        << "\\includegraphics[width=0.5\\textwidth,page=17]{digiValidationPlots.pdf}" << std::endl
//        << "\\hfill" << std::endl
//        << "\\includegraphics[width=0.5\\textwidth,page=18]{digiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  // GEM-CSC PadDigi validation plots

  file << "\\newpage" << std::endl;

  file << "\\section{GEM-CSC PadDigi validation plots}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{GEMCSCPadDigiValidationPlots.pdf} " << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl
       << std::endl;
  
  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=7]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=8]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=10]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=12]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=13]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  file << "\\begin{figure}[!htbp]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=15]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=16]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=17]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=18]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
//        << "\\\\" << std::endl
//        << "\\includegraphics[width=0.45\\textwidth,page=17]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
//        << "\\hfill" << std::endl
//        << "\\includegraphics[width=0.45\\textwidth,page=18]{GEMCSCPadDigiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;


  // GEM-CSC Coincidence PadDigi validation plots

  file << "\\newpage" << std::endl;

  file << "\\section{GEM-CSC Coincidence PadDigi validation plots}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{GEMCSCCoPadDigiValidationPlots.pdf} " << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=7]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=8]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl
       << std::endl;
  
  file << "\\begin{figure}[!htbp]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=10]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=15]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=17]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
//        << "\\\\" << std::endl
//        << "\\includegraphics[width=0.45\\textwidth,page=15]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
//        << "\\hfill" << std::endl
//        << "\\includegraphics[width=0.45\\textwidth,page=17]{GEMCSCCoPadDigiValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  // trackvalidation plots

  file << "\\newpage" << std::endl;

  file << "\\section{Matching efficiency plots}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{trackValidationPlots.pdf} " << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{trackValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl
       << std::endl;
  
  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=7]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=8]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=9]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=10]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=12]{trackValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=13]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=14]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=15]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=16]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=17]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=18]{trackValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=19]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=20]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=21]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=22]{trackValidationPlots.pdf}" << std::endl
//        << "\\\\" << std::endl
//        << "\\includegraphics[width=0.45\\textwidth,page=17]{trackValidationPlots.pdf}" << std::endl
//        << "\\hfill" << std::endl
//        << "\\includegraphics[width=0.45\\textwidth,page=18]{trackValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;
  
  file << "\\end{document}";
  
  file.close();
}

template<typename T> const std::string DigiValidator::to_string( T const& value )
{
  std::stringstream sstr;
  sstr << value;
  return sstr.str();
}
