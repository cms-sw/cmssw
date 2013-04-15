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
  const std::string ext_( process_.getParameter< std::string >( "ext" ) );
  const unsigned npads_(process_.getParameter< unsigned >( "npads" ));
  const std::string npadss(boost::lexical_cast< std::string >(npads_));  

  // Constants
  const std::string analyzer_("GEMDigiAnalyzer");
  std::string digis_("GEMDigiTree"); 
  std::string pads_("GEMCSCPadDigiTree"); 
  std::string copads_("GEMCSCCoPadDigiTree"); 
  std::string tracks_("TrackTree");

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

  TTree * treeDigis_( (TTree *) dirAna_->Get( digis_.c_str() ) );
  if ( ! treeDigis_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    digis '" << digis_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  TCut rm1("region==-1");
  TCut rp1("region==1");
  TCut l1("layer==1");
  TCut l2("layer==2");

  //  gStyle->SetOptStat( 1110 );
 
  /// XY occupancy plots
  TCanvas* c = new TCanvas("c","c",600,600);
  treeDigis_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rm1 && l1);
  TH2D* hh = (TH2D*)gDirectory->Get("hh");  
  hh->SetTitle("Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_globalxy_region-1_layer1" + ext_).c_str());

  c->Clear();
  treeDigis_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rm1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_globalxy_region-1_layer2" + ext_).c_str());

  c->Clear();
  treeDigis_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rp1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_globalxy_region1_layer1" + ext_).c_str());

  c->Clear();
  treeDigis_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rp1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1, layer2; globalX [cm]; globalY [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_globalxy_region1_layer2" + ext_).c_str());

  /// ZR occupancy plots
  c->Clear();		
  treeDigis_->Draw("g_r:g_z>>hh(200,-573,-564,55,130,240)",rm1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_globalzr_region-1" + ext_).c_str());

  c->Clear();		
  treeDigis_->Draw("g_r:g_z>>hh(200,564,573,55,130,240)",rp1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_globalzr_region1" + ext_).c_str());

  /// PhiStrip occupancy plots
  c->Clear();		
  treeDigis_->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)",rm1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1 layer1; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_phiStrip_region-1_layer1" + ext_).c_str());

  c->Clear();		
  treeDigis_->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)",rm1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region-1 layer2; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_phiStrip_region-1_layer2" + ext_).c_str());

  c->Clear();		
  treeDigis_->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)",rp1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1 layer1; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_phiStrip_region1_layer1" + ext_).c_str());
  
  c->Clear();		
  treeDigis_->Draw("strip:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312,192,0,384)",rp1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Digi occupancy: region1 layer2; phi [rad]; strip");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ +"strip_digi_phiStrip_region1_layer2" + ext_).c_str());

  /// Strip occupancy plots
  c->Clear();		
  treeDigis_->Draw("strip>>h(384,0.5,384.5)",rm1 && l1);
  TH1D* h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, region-1 layer1;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_region-1_layer1" + ext_).c_str());

  c->Clear();		
  treeDigis_->Draw("strip>>h(384,0.5,384.5)",rm1 && l2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, region-1 layer2;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_region-1_layer2" + ext_).c_str());

  c->Clear();		
  treeDigis_->Draw("strip>>h(384,0.5,384.5)",rp1 && l1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, region1 layer1;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_region1_layer1" + ext_).c_str());

  c->Clear();		
  treeDigis_->Draw("strip>>h(384,0.5,384.5)",rp1 && l2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Digi occupancy per strip number, region1 layer2;strip number;entries");		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_region1_layer2" + ext_).c_str());

  /// Bunch crossing plots
  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >(h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && l1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, layer1;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_layer1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && l2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, layer2; bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_layer2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && l1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, layer1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_layer1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && l2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, layer2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_layer2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_roll1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_roll2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_roll3" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_roll4" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_roll5" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region1_roll6" + ext_).c_str());  

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_roll1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_roll2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_roll3" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_roll4" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_roll5" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeDigis_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ +"strip_digi_bx_region-1_roll6" + ext_).c_str());  

//   gStyle->SetOptStat( 1110 );

  TTree * treePads_( (TTree *) dirAna_->Get( pads_.c_str() ) );
  if ( ! treePads_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    digis '" << pads_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  gPad->SetLogy(0);

  /// XY occupancy plots
  c->Clear();
  treePads_->Draw("g_x:g_y>>hh(1000,-260,260,1000,-260,260)",rm1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");  
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_globalxy_region-1_layer1" + ext_).c_str());

  treePads_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rm1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_globalxy_region-1_layer2" + ext_).c_str());

  c->Clear();
  treePads_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rp1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_globalxy_region1_layer1" + ext_).c_str());

  c->Clear();
  treePads_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rp1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region1, layer2; globalX [cm]; globalY [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_globalxy_region1_layer2" + ext_).c_str());

  /// ZR occupancy plots
  c->Clear();		
  treePads_->Draw("g_r:g_z>>hh(200,-573,-564,55,130,240)",rm1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region-1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_globalzr_region-1" + ext_).c_str());

  c->Clear();		
  treePads_->Draw("g_r:g_z>>hh(200,564,573,55,130,240)",rp1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_globalzr_region1" + ext_).c_str());

  /// Phi pad plots
  c->Clear();		
  treePads_->Draw(("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312," + npadss + ",0," + npadss + ")").c_str(),rm1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region-1 layer1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_phiPad_region-1_layer1" + ext_).c_str());

  c->Clear();		
  treePads_->Draw(("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312," + npadss + ",0," + npadss + ")").c_str(),rm1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region-1 layer2; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_phiPad_region-1_layer2" + ext_).c_str());

  c->Clear();		
  treePads_->Draw(("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312," + npadss + ",0," + npadss + ")").c_str(),rp1 && l1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region1 layer1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_phiPad_region1_layer1" + ext_).c_str());
  
  c->Clear();		
  treePads_->Draw(("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312," + npadss + ",0," + npadss + ")").c_str(),rp1 && l2);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("GEM-CSC Pad Digi occupancy: region1 layer2; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "pad_digi_phiPad_region1_layer2" + ext_).c_str());

  // Pad occupancy plots
  c->Clear();		
  treePads_->Draw(("pad>>h(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")").c_str());
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Pad occupancy: region-1 layer1;pad;entries/" + boost::lexical_cast< std::string >(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_region-1_layer1" + ext_).c_str());

  c->Clear();		
  treePads_->Draw(("pad>>h(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")").c_str(),rm1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Pad occupancy: region-1 layer2;pad;entries/" + boost::lexical_cast< std::string >(h->GetBinWidth(1)) + " pads").c_str());	
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_region-1_layer2" + ext_).c_str());

  c->Clear();		
  treePads_->Draw(("pad>>h(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")").c_str(),rp1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Pad occupancy: region1 layer1;pad;entries/" + boost::lexical_cast< std::string >(h->GetBinWidth(1)) + " pads").c_str());
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_region1_layer1" + ext_).c_str());

  c->Clear();		
  treePads_->Draw(("pad>>h(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")").c_str(),l1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Pad occupancy: region1 layer2;pad;entries/" + boost::lexical_cast< std::string >(h->GetBinWidth(1)) + " pads").c_str());
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_region1_layer2" + ext_).c_str());

  

  /// Bunch crossing plots
  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && l1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, layer1;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_layer1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && l2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, layer2; bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_layer2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && l1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, layer1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_layer1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && l2);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, layer2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_layer2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_roll1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_roll2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_roll3" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_roll4" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_roll5" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region1, roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region1_roll6" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_roll1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_roll2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_roll3" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_roll4" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_roll5" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treePads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Bunch crossing: region-1, roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "pad_digi_bx_region-1_roll6" + ext_).c_str());
  
  TTree * treeTracks_( (TTree *) dirAna_->Get( tracks_.c_str() ) );
  if ( ! treeTracks_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    digis '" << tracks_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

//   gStyle->SetOptStat( 0000 );
//   gStyle->SetOptFit( 0 );

  TCut etaMin("eta > 1.5");
  TCut etaMax("eta < 2.2");
  TCut etaCut(etaMin && etaMax);

  TCut simHitGEMl1("gem_sh_layer==1");
  TCut simHitGEMl2("gem_sh_layer==2");

  TCut simHitInOdd("has_gem_sh==1");
  TCut simHitInEven("has_gem_sh==2");
  TCut simHitInBoth("has_gem_sh==3");
  TCut atLeastOneSimHit(simHitInOdd || simHitInEven || simHitInBoth);

  TCut digiGEMl1("gem_dg_layer==1");
  TCut digiGEMl2("gem_dg_layer==2");

  TCut digiInOdd("has_gem_dg==1");
  TCut digiInEven("has_gem_dg==2");
  TCut digiInBoth("has_gem_dg==3");
  TCut atLeastOneDigi(digiInOdd || digiInEven || digiInBoth);

  TCut digiIn2Odd("has_gem_dg2==1");
  TCut digiIn2Even("has_gem_dg2==2");
  TCut digiIn2Both("has_gem_dg2==3");
  TCut twoDigi(digiIn2Odd || digiIn2Even || digiIn2Both);

  TCut padGEMl1("gem_pad_layer==1");
  TCut padGEMl2("gem_pad_layer==2");

  TCut padInOdd("has_gem_pad==1");
  TCut padInEven("has_gem_pad==2");
  TCut padInBoth("has_gem_pad==3");
  TCut atLeastOnePad(padInOdd || padInEven || padInBoth);

  TCut copadInOdd("has_gem_copad==1");
  TCut copadInEven("has_gem_copad==2");
  TCut copadInBoth("has_gem_copad==3");
  TCut atLeastOneCoPad(copadInOdd || copadInEven || copadInBoth);

  gPad->SetLogy(0);
  // Matching efficiency of SimTrack to Digi in GEML1
  c->Clear();
  gPad->SetGrid(1);  
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOneDigi && digiGEMl1);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl1);
  TH1D* g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_digi_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneDigi && digiGEMl1 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",digiGEMl1 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_digi_gem_layer1" + ext_ ).c_str());

  // Matching efficiency of SimTrack to Digi in GEML2
  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOneDigi && digiGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl2);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_digi_gem_layer2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneDigi && digiGEMl2 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",digiGEMl2 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_digi_gem_layer2" + ext_ ).c_str());

  // Matching efficiency of SimTrack to Digi in GEML1 and GEML2
  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",twoDigi);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 and GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_digi_gem_layer1and2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",twoDigi && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 and GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_digi_gem_layer1and2" + ext_ ).c_str());

  // Matching efficiency of SimTrack to Digi in GEML1 or GEML2
  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOneDigi && (digiGEMl1 || digiGEMl2));
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl1 || digiGEMl2);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 or GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_digi_gem_layer1or2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneDigi && (digiGEMl1 || digiGEMl2) && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",etaCut && (digiGEMl1 || digiGEMl2));
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 or GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_digi_gem_layer1or2" + ext_ ).c_str());

  // plots for matching efficiency of simtrack to digi with matched simhits
  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOneDigi && digiGEMl1 && atLeastOneSimHit && simHitGEMl1);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl1 && atLeastOneSimHit && simHitGEMl1);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_sim_digi_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneDigi && digiGEMl1 && atLeastOneSimHit && simHitGEMl1 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",digiGEMl1 && atLeastOneSimHit && simHitGEMl1 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl1 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_sim_digi_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOneDigi && digiGEMl2 && atLeastOneSimHit && simHitGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl2 && atLeastOneSimHit && simHitGEMl2);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Digi matching efficiency in GEMl2 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_sim_digi_gem_layer2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneDigi && digiGEMl2 && atLeastOneSimHit && simHitGEMl2 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",digiGEMl2 && atLeastOneSimHit && simHitGEMl2 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_sim_digi_gem_layer2" + ext_ ).c_str());

  // efficiency to match a simtrack to pad digi 
  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOnePad && digiGEMl1);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl1);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_pad_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOnePad && digiGEMl1 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",digiGEMl1 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_pad_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOnePad && digiGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",digiGEMl2);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_pad_gem_layer2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOnePad && digiGEMl2 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",digiGEMl2 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_pad_gem_layer2" + ext_ ).c_str());



  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOnePad);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 or GEMl2;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_pad_gem_layer1or2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOnePad && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 or GEMl2;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_pad_gem_layer1or2" + ext_ ).c_str());

  // pad digi plots with simhit selection
  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOnePad && padGEMl1 && atLeastOneSimHit && simHitGEMl1);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",padGEMl1 && atLeastOneSimHit && simHitGEMl1);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_sim_pad_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOnePad && padGEMl1 && atLeastOneSimHit && simHitGEMl1 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",padGEMl1 && atLeastOneSimHit && simHitGEMl1 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl1 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_sim_pad_gem_layer1" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOnePad && padGEMl2 && atLeastOneSimHit && simHitGEMl2);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)",padGEMl2 && atLeastOneSimHit && simHitGEMl2);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2 (with matched SimHits);#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_sim_pad_gem_layer2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOnePad && padGEMl2 && atLeastOneSimHit && simHitGEMl2 && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",padGEMl2 && atLeastOneSimHit && simHitGEMl2 && etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Pad matching efficiency in GEMl2 (with matched SimHits);#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_sim_pad_gem_layer2" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_eta>>h(100,1.5,2.2)",atLeastOneCoPad);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_eta>>g(100,1.5,2.2)");
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Co-Pad matching efficiency;#eta;Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_eta_tracks_copad_gem" + ext_ ).c_str());

  c->Clear();
  treeTracks_->Draw("gem_dg_phi>>h(100,-3.14159265358979312,3.14159265358979312)",atLeastOneCoPad && etaCut);
  h = (TH1D*)gDirectory->Get("h");
  treeTracks_->Draw("gem_dg_phi>>g(100,-3.14159265358979312,3.14159265358979312)",etaCut);
  g = (TH1D*)gDirectory->Get("g");
  for( int iBin=1; iBin< h->GetNbinsX()+1; ++iBin){
    h->SetBinContent(iBin,g->GetBinContent(iBin)==0 ? 0 : h->GetBinContent(iBin)/g->GetBinContent(iBin));
  }
  h->SetTitle("SimTrack to Co-Pad matching efficiency;#phi [rad];Efficiency");
  h->SetMinimum(0.);
  h->SetMaximum(1.1);
  h->Draw("");        
  c->SaveAs((targetDir_ + "eff_phi_tracks_copad_gem" + ext_ ).c_str());

  // GEM-CSC Coincidence pads  

  TTree * treeCoPads_( (TTree *) dirAna_->Get( copads_.c_str() ) );
  if ( ! treeCoPads_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    digis '" << copads_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  /// XY occupancy plots
  c->Clear();
  treeCoPads_->Draw("g_x:g_y>>hh(1000,-260,260,1000,-260,260)",rm1);
  hh = (TH2D*)gDirectory->Get("hh");  
  hh->SetTitle("Coincidence pad digi occupancy: region-1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "copad_digi_globalxy_region-1" + ext_).c_str());

  c->Clear();
  treeCoPads_->Draw("g_x:g_y>>hh(260,-260,260,260,-260,260)",rp1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Coincidence pad digi occupancy: region1; globalX [cm]; globalY [cm]");
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "copad_digi_globalxy_region1" + ext_).c_str());

  /// ZR occupancy plots
  c->Clear();		
  treeCoPads_->Draw("g_r:g_z>>hh(200,-568,-565,55,130,240)",rm1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Coincidence pad digi occupancy: region-1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "copad_digi_globalzr_region-1" + ext_).c_str());

  c->Clear();		
  treeCoPads_->Draw("g_r:g_z>>hh(200,565,568,55,130,240)",rp1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Coincidence pad digi occupancy: region1; globalZ [cm]; globalR [cm]");	
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "copad_digi_globalzr_region1" + ext_).c_str());

  /// Phi copad plots
  c->Clear();		
  treeCoPads_->Draw(("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312," + npadss + ",0," + npadss + ")").c_str(),rm1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Coincidence pad digi occupancy: region-1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "copad_digi_phiCopad_region-1" + ext_).c_str());

  c->Clear();		
  treeCoPads_->Draw(("pad:g_phi>>hh(280,-3.14159265358979312,3.14159265358979312," + npadss + ",0," + npadss + ")").c_str(),rp1);
  hh = (TH2D*)gDirectory->Get("hh");
  hh->SetTitle("Coincidence pad digi occupancy: region1; phi [rad]; pad");		
  hh->Draw("COLZ");
  c->SaveAs((targetDir_ + "copad_digi_phiCopad_region1" + ext_).c_str());

  // Pad occupancy plots
  c->Clear();		
  treeCoPads_->Draw(("pad>>h(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")").c_str(),rm1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Pad occupancy: region-1 layer1;pad;entries/" + boost::lexical_cast< std::string >(h->GetBinWidth(1)) + " pads").c_str());		
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_region-1" + ext_).c_str());

  c->Clear();		
  treeCoPads_->Draw(("pad>>h(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")").c_str(),rp1);
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle(("Pad occupancy: region-1 layer2;pad;entries/" + boost::lexical_cast< std::string >(h->GetBinWidth(1)) + " pads").c_str());	
  h->SetMinimum(0.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_region1" + ext_).c_str());

  /// Coincidence pad digi bunch crossing plots
  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region1, roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region1_roll1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region1, roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region1_roll2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region1, roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region1_roll3" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region1, roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region1_roll4" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region1, roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region1_roll5" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rp1 && "roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region1, roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region1_roll6" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==1");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region-1, roll1 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region-1_roll1" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==2");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region-1, roll2 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region-1_roll2" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==3");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region-1, roll3 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region-1_roll3" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==4");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region-1, roll4 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region-1_roll4" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==5");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region-1, roll5 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region-1_roll5" + ext_).c_str());

  c->Clear();		
  gPad->SetLogy();
  treeCoPads_->Draw("bx>>h(11,-5.5,5.5)",rm1 && "roll==6");
  h = (TH1D*)gDirectory->Get("h");
  h->SetTitle("Coincidence pad bunch crossing: region-1, roll6 ;bunch crossing;entries");			
  for( int uBin=1; uBin <= h->GetNbinsX(); ++uBin){
    h->GetXaxis()->SetBinLabel( uBin, boost::lexical_cast< std::string >( h->GetXaxis()->GetBinCenter(uBin) ).c_str() );
  }
  h->SetMinimum(1.);
  h->Draw("");
  c->SaveAs((targetDir_ + "copad_digi_bx_region-1_roll6" + ext_).c_str());

  gPad->SetLogy(0);
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
  

