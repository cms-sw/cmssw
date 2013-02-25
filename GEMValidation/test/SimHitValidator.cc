#include "SimHitValidator.h"

SimHitValidator::SimHitValidator()
{
  
}

SimHitValidator::~SimHitValidator()
{
  
}

void SimHitValidator::makeValidationPlots(const Selection& key = Muon)
{
  TString particleType[] = {"&&abs(particleType)==13","&&abs(particleType)!=13",""};
  TString titlePrefix[] = {"Muon","Non muon","All"};
  std::string fileSuffix[] = {"_muon","_nonmuon","_all"};
  
  std::cout << ">>> Opening TFile: " << getInFileName() << std::endl;  
  TFile *simHitFile = new TFile((getInFileName()).c_str());
  if (!simHitFile){
    std::cerr << "Error in SimHitValidator::produceValidationPlots() - no such TFile: " << getInFileName() << std::endl; 
    return;
  }    
  
  std::cout << ">>> Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
  TDirectory * dir = (TDirectory*)simHitFile->Get("gemSimHitAnalyzer"); 
  if (!dir){
    std::cerr << ">>> Error in SimHitValidator::produceValidationPlots() - No such TDirectory: gemSimHitAnalyzer" << std::endl;
    return;
  }

  std::cout << ">>> Reading TTree: GEMSimHits" << std::endl;
  TTree* tree = (TTree*) dir->Get("GEMSimHits");
  if (!tree){
    std::cerr << ">>> Error in SimHitValidator::produceValidationPlots() - No such TTree: GEMSimHits" << std::endl;
    return;
  }

  std::cout << ">>> Producing PDF file: " << "simhitValidationPlots" + fileSuffix[(int)key] + ".pdf" << std::endl;

  ////////////////////////
  // XY occupancy plots //
  ////////////////////////
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
  
  ////////////////////////
  // ZR occupancy plots //
  ////////////////////////
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
  
  ////////////////////////
  // timeOfFlight plots //
  ////////////////////////
  c->Clear();
  tree->Draw("timeOfFlight>>h(40,18,22)","region==-1&&layer==1" + particleType[(int)key]);
  TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region-1_layer1");
  
  c->Clear();
  tree->Draw("timeOfFlight>>h(40,18,22)","region==-1&&layer==2" + particleType[(int)key]);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region-1_layer2");
  
  c->Clear();
  tree->Draw("timeOfFlight>>h(40,18,22)","region==1&&layer==1" + particleType[(int)key]);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region1_layer1");

  c->Clear();
  tree->Draw("timeOfFlight>>h(40,18,22)","region==1&&layer==2" + particleType[(int)key]);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries" );
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:timeOfFlight_region1_layer2");

  ///////////////////
  // momentum plot //
  ///////////////////
  c->Clear();
  tree->Draw("pabs>>h(200,0.,100.)",""+(particleType[(int)key])(2,particleType[(int)key].Length()));
  h = (TH1D*)gDirectory->Get("h");
  gPad->SetLogx(0);
  h->SetTitle( titlePrefix[(int)key] + " SimHits absolute momentum;Momentum [GeV/c];entries" );       
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:momentum");

  //////////////////
  // pdg ID plots //
  //////////////////
  c->Clear();
  gPad->SetLogx();
  tree->Draw("(particleType>0)?TMath::Log10(particleType):-TMath::Log10(particleType)>>h(200,-100.,100.)","");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle( titlePrefix[(int)key] + " SimHit PDG Id;PDG Id;entries"  );       
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:pdgid");

  ////////////////////////
  // eta occupancy plot //
  ////////////////////////
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
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf").c_str(),"Title:globalEta");

  //////////////////////
  // energy loss plot //
  //////////////////////
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
  c->SaveAs(("simhitValidationPlots" + fileSuffix[(int)key] + ".pdf)").c_str(),"Title:muon_energy_plot");
}

void SimHitValidator::makeTrackValidationPlots()
{
  std::cout << ">>> Opening TFile: " << getInFileName() << std::endl;  
  TFile *simHitFile = new TFile((getInFileName()).c_str());
  if (!simHitFile){  
    std::cerr << "Error in SimHitValidator::makeTrackValidationPlots() - no such TFile: " << getInFileName() << std::endl; 
    return;
  }    
  
  std::cout << ">>> Opening TDirectory: gemSimHitAnalyzer" << std::endl;  
  TDirectory * dir = (TDirectory*)simHitFile->Get("gemSimHitAnalyzer"); 
  if (!dir){
    std::cerr << ">>> Error in SimHitValidator::makeTrackValidationPlots() - No such TDirectory: gemSimHitAnalyzer" << std::endl;
    return;
  }

  std::cout << ">>> Reading TTree: Tracks" << std::endl;
  TTree* tree = (TTree*) dir->Get("Tracks");
  if (!tree){
    std::cerr << ">>> Error in SimHitValidator::makeTrackValidationPlots() - No such TTree: Tracks" << std::endl;
    return;
  }

  std::cout << ">>> Producing PDF file: " << "trackValidationPlots.pdf" << std::endl;

  // efficiency in eta of matching a track to simhits in layer 1
  TCanvas* c = new TCanvas("c","c",600,600);
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_sh==1||has_gem_sh==2||has_gem_sh==3)&&gem_sh_layer==1");
  TH1D* h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_sh_layer==1");
  TH1D* g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf(","Title:eff_eta_tracks_simhit_gem_layer1");

  // efficiency in phi of matching a track to simhits in layer 1
  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312)","(has_gem_sh==1||has_gem_sh==2||has_gem_sh==3)&&gem_sh_layer==1");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312)","gem_sh_layer==1");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_simhit_gem_layer1");

  // efficiency in eta of matching a track to simhits in layer 2
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_sh==1||has_gem_sh==2||has_gem_sh==3)&&gem_sh_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)","gem_sh_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_simhit_gem_layer2");

  // efficiency in phi of matching a track to simhits in layer 2
  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312)","(has_gem_sh==1||has_gem_sh==2||has_gem_sh==3)&&gem_sh_layer==2");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312)","gem_sh_layer==2");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_phi_tracks_simhit_gem_layer2");

  // efficiency in eta of matching a track to simhits in layer 1 and 2
  c->Clear();
  tree->Draw("eta>>h(100,1.6,2.1)","(has_gem_sh2==1||has_gem_sh2==2||has_gem_sh2==3)");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("eta>>g(100,1.6,2.1)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1 and GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf","Title:eff_eta_tracks_simhit_gem_layer12");

  // efficiency in phi of matching a track to simhits in layer 1 and 2
  c->Clear();
  tree->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312)","(has_gem_sh2==1||has_gem_sh2==2||has_gem_sh2==3)");
  h = (TH1D*)gDirectory->Get("h");
  tree->Draw("phi>>g(100,-3.14159265358979312,3.14159265358979312)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1 and GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs("trackValidationPlots.pdf)","Title:eff_phi_tracks_simhit_gem_layer12");





//   //////////////////////////
//   // SimHit GEML1 - GEML2 //
//   //////////////////////////
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even>>h(100,-0.0005,0.0005)",
// 	     "gem_sh_layer==1");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Even;#Delta#eta(GEMl1,GEMl2);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf(","Title:delta_meanSimHitEtaGEMl1Even-meanSimHitEtaGEMl2Even");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even>>h(100,-0.001,0.001)",
// 	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoGEMl2Even>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Even;#Delta#phi(GEMl1,GEMl2);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even-meanSimHitPhiGEMl2Even");
  
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd>>h(100,-0.0005,0.0005)",
// 	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Odd;#Delta#eta(GEMl1,GEMl2);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaGEMl2Odd");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd>>h(100,-0.001,0.001)",
// 	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoGEMl2Odd>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Odd;#Delta#phi(GEMl1,GEMl2);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiGEMl2Odd");
 
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both>>h(100,-0.0005,0.0005)",
// 	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl1-GEMl2 Both;#Delta#eta(GEMl1,GEMl2);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both-meanSimHitEtaGEMl2Both");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both>>h(100,-0.001,0.001)",
// 	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoGEMl2Both>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl1-GEMl2 Both;#Delta#phi(GEMl1,GEMl2);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both-meanSimHitPhiGEMl2Both");

//   ////////////////////////
//   // simhit GEML1 - CSC //
//   ////////////////////////
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven>>h(100,-0.003,0.003)",
// 	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl1-CSC Even;#Delta#eta(GEMl1,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Even-meanSimHitEtaCSCEven");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven>>h(100,-0.005,0.005)",
// 	     "meanSimHitRhoGEMl1Even>0 && meanSimHitRhoCSCEven>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl1-CSC Even;#Delta#phi(GEMl1,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Even-meanSimHitPhiCSCEven");

//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd>>h(100,-0.005,0.005)",
// 	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl1-CSC Odd;#Delta#eta(GEMl1,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Odd-meanSimHitEtaCSCOdd");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd>>h(100,-0.007,0.007)",
// 	     "meanSimHitRhoGEMl1Odd>0 && meanSimHitRhoCSCOdd>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl1-CSC Odd;#Delta#phi(GEMl1,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Odd-meanSimHitPhiCSCOdd");
 
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth>>h(100,-0.005,0.005)",
// 	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl1-CSC Both;#Delta#eta(GEMl1,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl1Both-meanSimHitEtaCSCBoth");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth>>h(100,-0.007,0.007)",
// 	     "meanSimHitRhoGEMl1Both>0 && meanSimHitRhoCSCBoth>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl1-CSC Both;#Delta#phi(GEMl1,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl1Both-meanSimHitPhiCSCBoth");

//   ////////////////////////
//   // simhit GEML2 - CSC //
//   ////////////////////////
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven>>h(100,-0.002,0.002)",
// 	     "meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl2-CSC Even;#Delta#eta(GEMl2,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Even-meanSimHitEtaCSCEven");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven>>h(100,-0.003,0.003)",
// 	     "meanSimHitRhoGEMl2Even>0 && meanSimHitRhoCSCEven>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Even;#Delta#phi(GEMl2,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Even-meanSimHitPhiCSCEven");

//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd>>h(100,-0.004,0.004)",
// 	     "meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl2-CSC Odd;#Delta#eta(GEMl2,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Odd-meanSimHitEtaCSCOdd");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd>>h(100,-0.01,0.01)",
// 	     "meanSimHitRhoGEMl2Odd>0 && meanSimHitRhoCSCOdd>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Odd;#Delta#phi(GEMl2,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitPhiGEMl2Odd-meanSimHitPhiCSCOdd");
  
//   c->Clear();
//   tree->Draw("meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth>>h(100,-0.004,0.004)",
// 	     "meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
//   h = (TH1D*)gDirectory->Get("h");
//   h->SetTitle( "Delta Eta SimHit GEMl2-CSC Both;#Delta#eta(GEMl2,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf","Title:delta_meanSimHitEtaGEMl2Both-meanSimHitEtaCSCBoth");

//   c->Clear();
//   tree->Draw("meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth>>h(100,-0.008,0.008)",
// 	     "meanSimHitRhoGEMl2Both>0 && meanSimHitRhoCSCBoth>0");
//   h = (TH1D*)gDirectory->Get("h");

//   h->SetTitle( "Delta Phi SimHit GEMl2-CSC Both;#Delta#phi(GEMl2,CSC);Entries" );       
//   h->Draw("");        
//   c->SaveAs("deltaTrackValidationPlots.pdf)","Title:delta_meanSimHitPhiGEMl2Both-meanSimHitPhiCSCBoth");

}

void SimHitValidator::makeValidationReport()
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
       << "\\item[Dataset path:] {\\scriptsize \\texttt{" << dsp << "}}"  << std::endl
       << "\\item[Dataset size:] " << getDataSetSize() << " - " << getNumberOfEvents() << std::endl
       << "\\item[Time to complete:] " << getTimeToComplete() << std::endl
       << "\\item[Number of jobs:] " << getNumberOfJobs() << std::endl
       << "\\item[Estimated time/event:] " << getNumberOfEvents() << std::endl
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

  file << "\\section{SimHit validation plots}" << std::endl;

  // GlobalY versus GlobalX
  
  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{simhitValidationPlots_all.pdf} " << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\end{figure}" << std::endl
       << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  // GlobalR versus GlobalZ

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{simhitValidationPlots_all.pdf} " << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\end{figure}" << std::endl
       << std::endl;

  // Absolute momentum and energy loss

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=14]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=14]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=11]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=14]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  // PDG ID and eta occupancy

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=12]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=13]{simhitValidationPlots_all.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=12]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=13]{simhitValidationPlots_muon.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=12]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=13]{simhitValidationPlots_nonmuon.pdf}" << std::endl
       << "\\end{figure}" << std::endl;

  // Matching efficiency plots

  file << "\\section{Matching efficiency plots}" << std::endl;

  file << "\\begin{figure}[h!]" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=1]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=2]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=3]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=4]{trackValidationPlots.pdf}" << std::endl
       << "\\\\" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=5]{trackValidationPlots.pdf}" << std::endl
       << "\\hfill" << std::endl
       << "\\includegraphics[width=0.45\\textwidth,page=6]{trackValidationPlots.pdf}" << std::endl
       << "\\end{figure}" << std::endl;
  
	    
  file << "\\end{document}" << std::endl;
  
  file.close();
}


void SimHitValidator::setEtaBinLabels(const TH1D* h)
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
