#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>

#include "boost/lexical_cast.hpp"

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TKey.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TF2.h>
#include <TFitResult.h>
#include <TMath.h>
#include <TCanvas.h>
#include <TCut.h>

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

int main( int argc, char * argv[] )
{
  int returnStatus_( 0 );
  
  // Load libraries
  gSystem->Load( "libFWCoreFWLite" );
  AutoLibraryLoader::enable();
  

  // Check configuration file
  if ( argc < 2 ) {
    std::cout << argv[ 0 ] << " --> Usage:" << std::endl
              << "    " << argv[ 0 ] << " [CONFIG_FILE.py]" << std::endl;
    returnStatus_ += 0x1;
    return returnStatus_;
  }
  
  if ( ! edm::readPSetsFrom( argv[ 1 ] )->existsAs< edm::ParameterSet >( "process" ) ) {
    std::cout << argv[ 0 ] << " --> ERROR:" << std::endl
              << "    cms.PSet 'process' missing in " << argv[ 1 ] << std::endl;
    returnStatus_ += 0x2;
    return returnStatus_;
  }
  
  const edm::ParameterSet & process_( edm::readPSetsFrom( argv[ 1 ] )->getParameter< edm::ParameterSet >( "process" ) );
  const unsigned verbose_( process_.getParameter< unsigned >( "verbose" ) );
  const std::string inputFile_( process_.getParameter< std::string >( "inputFile" ) );
  const std::string targetDir_( process_.getParameter< std::string >( "targetDir" ) );
  const bool printFile_( process_.getParameter< bool >( "printFile" ) );
  const std::vector<std::string> muonSelection_( process_.getParameter< std::vector<std::string> >( "muonSelection" ) );
  const std::vector<std::string> titlePrefix_( process_.getParameter< std::vector<std::string> >( "titlePrefix" ) );
  const std::vector<std::string> histSuffix_( process_.getParameter< std::vector<std::string> >( "histSuffix" ) );
  const std::string ext_( process_.getParameter< std::string >( "ext" ) );
  const int nregion_( process_.getParameter< int >( "nregion" ) );
  const int nlayer_( process_.getParameter< int >( "nlayer" ) );
  const int npart_( process_.getParameter< int >( "npart" ) );

  // Constants
  const std::string analyzer_( "GEMSimHitAnalyzer" );
  std::string simHits_( "GEMSimHits" ); 
  std::string simTracks_( "Tracks" ); 
  
  // Open input file
  if ( verbose_ > 0 )
    std::cout << std::endl
              << argv[ 0 ] << " --> INFO:" << std::endl
              << "    using      input  file '" << inputFile_  << "'" << std::endl;
  
  TFile * fileIn_( TFile::Open( inputFile_.c_str(), "UPDATE" ) );
  if ( ! fileIn_ ) {
    std::cout << argv[ 0 ] << " --> ERROR:" << std::endl
              << "    input file '" << inputFile_ << "' missing" << std::endl;
    returnStatus_ += 0x10;
    return returnStatus_;
  }

  TDirectory * dirAna_( (TDirectory *) fileIn_->Get( analyzer_.c_str() ) );
  if ( ! dirAna_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    simhits '" << analyzer_ << "' does not exist in input file" << std::endl;
    returnStatus_ += 0x20;
    return returnStatus_;
  }

  TTree * treeHits_( (TTree *) dirAna_->Get( simHits_.c_str() ) );
  if ( ! treeHits_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    simhits '" << simHits_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  const TCut rm1("region==-1");
  const TCut rp1("region==1");
  const TCut l1("layer==1");
  const TCut l2("layer==2");
  //  gStyle->SetOptStat( 1110 );

  for (unsigned uSel = 0; uSel < 3; ++uSel){
    
    /// XY occupancy plots
    TCanvas* c = new TCanvas("c","c",600,600);
    treeHits_->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)",rm1 && l1 && muonSelection_.at(uSel).c_str());
    TH2D *hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle((titlePrefix_.at(uSel) + " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]").c_str());
    hh->Draw("COLZ");    
    c->SaveAs((targetDir_ + "simhit_globalxy_region-1_layer1" + histSuffix_.at(uSel) + ".png").c_str());
    
    c->Clear();
    treeHits_->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)",rm1 && l2 && muonSelection_.at(uSel).c_str());
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle((titlePrefix_.at(uSel) + " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]").c_str());   
    hh->Draw("COLZ");      
    c->SaveAs((targetDir_ +"simhit_globalxy_region-1_layer2" + histSuffix_.at(uSel) + ".png").c_str());
    
    c->Clear();
    treeHits_->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)",rp1 && l1 && muonSelection_.at(uSel).c_str());
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle((titlePrefix_.at(uSel) + " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]").c_str());   
    hh->Draw("COLZ");    
    c->SaveAs((targetDir_ +"simhit_globalxy_region1_layer1" + histSuffix_.at(uSel) + ".png").c_str());
    
    c->Clear();
    treeHits_->Draw("globalY:globalX>>hh(100,-260,260,100,-260,260)",rp1 && l2 && muonSelection_.at(uSel).c_str());
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle((titlePrefix_.at(uSel) + " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]").c_str());   
    hh->Draw("COLZ");    
    c->SaveAs((targetDir_ +"simhit_globalxy_region1_layer2" + histSuffix_.at(uSel) + ".png").c_str());

    /// ZR occupancy plots
    c->Clear();
    treeHits_->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,-573,-564,110,130,240)",rm1 && muonSelection_.at(uSel).c_str());
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle((titlePrefix_.at(uSel) + " SimHit occupancy: region-1;globalZ [cm];globalR [cm]").c_str());
    hh->Draw("COLZ");    
    c->SaveAs((targetDir_ +"simhit_globalzr_region-1" + histSuffix_.at(uSel) + ".png").c_str());
  
    c->Clear();
    treeHits_->Draw("sqrt(globalX*globalX+globalY*globalY):globalZ>>hh(200,564,573,110,130,240)",rp1 && muonSelection_.at(uSel).c_str());
    hh = (TH2D*)gDirectory->Get("hh");
    hh->SetTitle((titlePrefix_.at(uSel) + " SimHit occupancy: region1;globalZ [cm];globalR [cm]").c_str());
    hh->Draw("COLZ");    
    c->SaveAs((targetDir_ +"simhit_globalzr_region1" + histSuffix_.at(uSel) + ".png").c_str());

    /// timeOfFlight plots 
    c->Clear();
    treeHits_->Draw("timeOfFlight>>h(40,18,22)",rm1 && l1 && muonSelection_.at(uSel).c_str());
    TH1D* h = (TH1D*)gDirectory->Get("h");
    h->SetTitle((titlePrefix_.at(uSel) + " SimHit timeOfFlight: region-1, layer1;Time of flight [ns];entries").c_str());
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_timeOfFlight_region-1_layer1" + histSuffix_.at(uSel) + ".png").c_str());
    
    c->Clear();
    treeHits_->Draw("timeOfFlight>>h(40,18,22)",rm1 && l2 && muonSelection_.at(uSel).c_str());
    h = (TH1D*)gDirectory->Get("h");
    h->SetTitle((titlePrefix_.at(uSel) + " SimHit timeOfFlight: region-1, layer2;Time of flight [ns];entries").c_str());
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_timeOfFlight_region-1_layer2" + histSuffix_.at(uSel) + ".png").c_str());
  
    c->Clear();
    treeHits_->Draw("timeOfFlight>>h(40,18,22)",rp1 && l1 && muonSelection_.at(uSel).c_str());
    h = (TH1D*)gDirectory->Get("h");
    h->SetTitle((titlePrefix_.at(uSel) + " SimHit timeOfFlight: region1, layer1;Time of flight [ns];entries").c_str());
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_timeOfFlight_region1_layer1" + histSuffix_.at(uSel) + ".png").c_str());
    
    c->Clear();
    treeHits_->Draw("timeOfFlight>>h(40,18,22)",rp1 && l2 && muonSelection_.at(uSel).c_str());
    h = (TH1D*)gDirectory->Get("h");
    h->SetTitle((titlePrefix_.at(uSel) + " SimHit timeOfFlight: region1, layer2;Time of flight [ns];entries").c_str());
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_timeOfFlight_region1_layer2" + histSuffix_.at(uSel) + ".png").c_str());

    /// momentum plot
    c->Clear();
    treeHits_->Draw("pabs>>h(200,0.,200.)",muonSelection_.at(uSel).c_str());
    h = (TH1D*)gDirectory->Get("h");
    gPad->SetLogx(0);
    gPad->SetLogy(1);
    h->SetTitle((titlePrefix_.at(uSel) + " SimHits absolute momentum;Momentum [GeV/c];entries").c_str());       
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_momentum" + histSuffix_.at(uSel) + ".png").c_str());

    /// pdg ID plots
    c->Clear();
    //   gPad->SetLogx();
    //  treeHits_->Draw("((particleType>0)?TMath::Log10(particleType):-TMath::Log10(-particleType))>>h(200,-100.,100.)",muonSelection_.at(uSel).c_str());
    // ""+(muonSelection_.at(uSel).c_str()).substr(2,muonSelection_.at(uSel).c_str().Length())
    treeHits_->Draw("particleType>>h(200,-100.,100.)",muonSelection_.at(uSel).c_str());
    h = (TH1D*)gDirectory->Get("h");
    h->SetTitle((titlePrefix_.at(uSel) + " SimHit PDG Id;PDG Id;entries").c_str());       
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_pdgid" + histSuffix_.at(uSel) + ".png").c_str());

    /// eta occupancy plot
    int region=0;
    int layer=0;
    int roll=0;  
    int particletype=0;
    TBranch *b_region;
    TBranch *b_layer;
    TBranch *b_roll;
    TBranch *b_particleType;  
    treeHits_->SetBranchAddress("region", &region, &b_region);
    treeHits_->SetBranchAddress("layer", &layer, &b_layer);
    treeHits_->SetBranchAddress("roll", &roll, &b_roll);
    treeHits_->SetBranchAddress("particleType", &particletype, &b_particleType);
    h = new TH1D("h", (titlePrefix_.at(uSel) + " occupancy in eta partitions; occupancy in #eta partition; entries").c_str(),4*npart_,1.,1.+4*npart_);
    int nbytes = 0;
    int nb = 0;
    for (Long64_t jentry=0; jentry<treeHits_->GetEntriesFast();jentry++) {
      Long64_t ientry = treeHits_->LoadTree(jentry);
      if (ientry < 0) break;
      nb = treeHits_->GetEntry(jentry);   
      nbytes += nb;
      switch(uSel){
      case 0:
	if (abs(particletype)==13) h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) );
	break;
      case 1:
	if (abs(particletype)!=13) h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) );
	break;
      case 2:
	h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) );
	break;
      }
    }    
    c->Clear();  
    gPad->SetLogx(0);
    gPad->SetLogy(0);
    int ibin(1);
    for (int iregion = 1; iregion<nregion_+1; ++iregion){
      TString region( (iregion == 1) ? "-" : "+" );
      for (int ilayer = 1; ilayer<nregion_+1; ++ilayer){
	TString layer( TString::Itoa(ilayer,10)); 
	for (int ipart = 1; ipart<npart_+1; ++ipart){
	  TString part( TString::Itoa(ipart,10)); 
	  h->GetXaxis()->SetBinLabel(ibin,region+layer+part);
	  ++ibin;
	}
      }
    }
    
    h->SetMinimum(0.);
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_globalEta" + histSuffix_.at(uSel) + ".png").c_str());
    
    /// energy loss plot
    h = new TH1D("h","",60,0.,6000.);
    Float_t energyLoss=0;
    TBranch *b_energyLoss;
    treeHits_->SetBranchAddress("energyLoss", &energyLoss, &b_energyLoss);
    for (Long64_t jentry=0; jentry<treeHits_->GetEntriesFast();jentry++) {
      Long64_t ientry = treeHits_->LoadTree(jentry);
      if (ientry < 0) break;
      nb = treeHits_->GetEntry(jentry);   
      nbytes += nb;
      switch(uSel){
      case 0:
	if (abs(particletype)==13) h->Fill( energyLoss*1.e9 );
	break;
      case 1:
	if (abs(particletype)!=13) h->Fill( energyLoss*1.e9 );
      break;
      case 2:
	h->Fill( energyLoss*1.e9 );
	break;
      }
    }
    c->Clear();  
    gPad->SetLogx(0);
    gPad->SetLogy(0);
    h->SetTitle((titlePrefix_.at(uSel) + " SimHit energy loss;Energy loss [eV];entries").c_str());
    h->SetMinimum(0.);
    h->Draw("");        
    c->SaveAs((targetDir_ +"simhit_energyloss" + histSuffix_.at(uSel) + ".png").c_str());
  }
  
  TTree * treeTracks_( (TTree *) dirAna_->Get( simTracks_.c_str() ) );
  if ( ! treeTracks_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    tracks '" << simTracks_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  TCut etaMin = "eta > 1.5";
  TCut etaMax = "eta < 2.2";
  TCut etaCut = etaMin && etaMax;

  TCut simHitGEMl1 = "gem_sh_layer==1";
  TCut simHitGEMl2 = "gem_sh_layer==2";

  TCut simHitInOdd = "has_gem_sh==1";
  TCut simHitInEven = "has_gem_sh==2";
  TCut simHitInBoth = "has_gem_sh==3";
  TCut atLeastOneSimHit = simHitInOdd || simHitInEven || simHitInBoth;

  TCut simHitIn2Odd = "has_gem_sh2==1";
  TCut simHitIn2Even = "has_gem_sh2==2";
  TCut simHitIn2Both = "has_gem_sh2==3";
  TCut twoSimHits = simHitIn2Odd || simHitIn2Even || simHitIn2Both;
  //  gStyle->SetOptStat( 0000 );

  // efficiency in eta of matching a track to simhits in layer 1
  TCanvas* c = new TCanvas("c","c",600,600);
  c->Clear();
  gPad->SetGrid(1);  
  treeTracks_->Draw("gem_sh_eta>>h(100,1.5,2.2)",atLeastOneSimHit && simHitGEMl1);
  TH1D* h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_sh_eta>>g(100,1.5,2.2)",simHitGEMl1);
  TH1D* g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1;SimHit #eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.05);
  h->Draw("");        
  c->SaveAs((targetDir_ +"eff_eta_tracks_simhit_gem_layer1.png").c_str());

  // efficiency in phi of matching a track to simhits in layer 1
  c->Clear();
  treeTracks_->Draw("gem_sh_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneSimHit && simHitGEMl1 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_sh_phi>>g(100,-3.14159265358979312,3.14159265358979312)",simHitGEMl1 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1;SimHit #phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.05);
  h->Draw("");        
  c->SaveAs((targetDir_ +"eff_phi_tracks_simhit_gem_layer1.png").c_str());

  // efficiency in eta of matching a track to simhits in layer 2
  c->Clear();
  treeTracks_->Draw("gem_sh_eta>>h(100,1.5,2.2)",atLeastOneSimHit && simHitGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_sh_eta>>g(100,1.5,2.2)",simHitGEMl2);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl2;SimHit #eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.05);
  h->Draw("");        
  c->SaveAs((targetDir_ +"eff_eta_tracks_simhit_gem_layer2.png").c_str());

  // efficiency in phi of matching a track to simhits in layer 2
  c->Clear();
  treeTracks_->Draw("gem_sh_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneSimHit && simHitGEMl2 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_sh_phi>>g(100,-3.14159265358979312,3.14159265358979312)",simHitGEMl2 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl2;SimHit #phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.05);
  h->Draw("");        
  c->SaveAs((targetDir_ +"eff_phi_tracks_simhit_gem_layer2.png").c_str());

  // efficiency in eta of matching a track to simhits in layer 1 and 2
  c->Clear();
  treeTracks_->Draw("gem_sh_eta>>h(100,1.5,2.2)",twoSimHits);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_sh_eta>>g(100,1.5,2.2)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1 and GEMl2;SimHit #eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.05);
  h->Draw("");        
  c->SaveAs((targetDir_ +"eff_eta_tracks_simhit_gem_layer12.png").c_str());

  // efficiency in phi of matching a track to simhits in layer 1 and 2
  c->Clear();
  treeTracks_->Draw("gem_sh_phi>>h(100,-3.14159265358979312,3.14159265358979312)",twoSimHits && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_sh_phi>>g(100,-3.14159265358979312,3.14159265358979312)",etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("Efficiency of matching a SimTrack to SimHits in GEMl1 and GEMl2;SimHit #phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.05);
  h->Draw("");        
  c->SaveAs((targetDir_ +"eff_phi_tracks_simhit_gem_layer12.png").c_str());
  
  c->Clear();
  treeTracks_->Draw("eta-gem_sh_eta>>h(100,-0.1,0.1)",simHitGEMl1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Delta #eta track-simhit GEMl1;#Delta #eta;Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"simhit_delta_eta_layer1.png").c_str());

  c->Clear();
  treeTracks_->Draw("phi-gem_sh_phi>>h(100,-0.1,0.1)",simHitGEMl1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Delta #phi track-simhit GEMl1;#Delta #phi;Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"simhit_delta_phi_layer1.png").c_str());

  c->Clear();
  treeTracks_->Draw("eta-gem_sh_eta>>h(100,-0.1,0.1)",simHitGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Delta #eta track-simhit GEMl2;#Delta #eta;Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"simhit_delta_eta_layer2.png").c_str());

  c->Clear();
  treeTracks_->Draw("phi-gem_sh_phi>>h(100,-0.1,0.1)",simHitGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Delta #phi track-simhit GEMl2;#Delta #phi;Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"simhit_delta_phi_layer2.png").c_str());

  /// Track kinematics plots
  c->Clear();
  treeTracks_->Draw("pt>>h(100,0,200)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Track p_{T};Track p_{T} [GeV];Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"track_pt.png").c_str());

  c->Clear();
  treeTracks_->Draw("eta>>h(100,1.5,2.2)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Track #eta;Track #eta;Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"track_eta.png").c_str());

  c->Clear();
  treeTracks_->Draw("phi>>h(100,-3.14159265358979312,3.14159265358979312)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Track #phi;Track #phi [rad];Entries");
  h->SetMinimum(0.);
  h->Draw("");        
  c->SaveAs((targetDir_ +"track_phi.png").c_str());

  return returnStatus_;
}
  

