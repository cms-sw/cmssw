#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iterator>

#include "boost/lexical_cast.hpp"

#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TStyle.h>
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
#include <TGraphAsymmErrors.h>
#include <TPaveStats.h>
#include <TText.h>

#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

void draw_geff(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	       TString to_draw, TCut denom_cut, TCut extra_num_cut, TString opt = "", int color = kBlue, int marker_st = 1, float marker_sz = 1.);
void draw_occ(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	      TString to_draw, TCut cut, TString opt = "");
void draw_1D(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	     TString to_draw, TCut cut, TString opt = "");
void draw_bx(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	     TString to_draw, TCut cut, TString opt = "");

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
  const TString inputFile_( process_.getParameter< std::string >( "inputFile" ) );
  const TString targetDir_( process_.getParameter< std::string >( "targetDir" ) );
  const TString ext_( process_.getParameter< std::string >( "ext" ) );
  const unsigned npads_(process_.getParameter< unsigned >( "npads" ));
  const TString npadss(boost::lexical_cast< std::string >(npads_));  

  // Constants
  const TString analyzer_("GEMDigiAnalyzer");
  const TString digis_("GEMDigiTree"); 
  const TString pads_("GEMCSCPadDigiTree"); 
  const TString copads_("GEMCSCCoPadDigiTree"); 
  const TString tracks_("TrackTree");
  const TCut rm1("region==-1");
  const TCut rp1("region==1");
  const TCut l1("layer==1");
  const TCut l2("layer==2");

  // Open input file
  std::cout << std::endl
	    << argv[ 0 ] << " --> INFO:" << std::endl
	    << "    using      input  file '" << inputFile_  << "'" << std::endl;
  
  TFile * fileIn_( TFile::Open( inputFile_, "UPDATE" ) );
  if ( ! fileIn_ ) {
    std::cout << argv[ 0 ] << " --> ERROR:" << std::endl
              << "    input file '" << inputFile_ << "' missing" << std::endl;
    returnStatus_ += 0x10;
    return returnStatus_;
  }

  TDirectory * dirAna_( (TDirectory *) fileIn_->Get( analyzer_ ) );
  if ( ! dirAna_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    simhits '" << analyzer_ << "' does not exist in input file" << std::endl;
    returnStatus_ += 0x20;
    return returnStatus_;
  }

  TTree * treeDigis_( (TTree *) dirAna_->Get( digis_ ) );
  if ( ! treeDigis_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    digis '" << digis_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  /// occupancy plots
  draw_occ(targetDir_, "strip_dg_xy_rm1_l1", ext_, treeDigis_, "Digi occupancy: region-1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1 && l1, "COLZ");
  draw_occ(targetDir_, "strip_dg_xy_rm1_l2", ext_, treeDigis_, "Digi occupancy: region-1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1 && l2, "COLZ");
  draw_occ(targetDir_, "strip_dg_xy_rp1_l1", ext_, treeDigis_, "Digi occupancy: region1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1 && l1, "COLZ");
  draw_occ(targetDir_, "strip_dg_xy_rp1_l2", ext_, treeDigis_, "Digi occupancy: region1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1 && l2, "COLZ"); 

  draw_occ(targetDir_, "strip_dg_zr_rm1", ext_, treeDigis_, "Digi occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ");
  draw_occ(targetDir_, "strip_dg_zr_rp1", ext_, treeDigis_, "Digi occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ");

  draw_occ(targetDir_, "strip_dg_phistrip_rm1_l1", ext_, treeDigis_, "Digi occupancy: region-1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", rm1 && l1, "COLZ");
  draw_occ(targetDir_, "strip_dg_phistrip_rm1_l2", ext_, treeDigis_, "Digi occupancy: region-1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", rm1 && l2, "COLZ");
  draw_occ(targetDir_, "strip_dg_phistrip_rp1_l1", ext_, treeDigis_, "Digi occupancy: region1 layer1; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", rp1 && l1, "COLZ");
  draw_occ(targetDir_, "strip_dg_phistrip_rp1_l2", ext_, treeDigis_, "Digi occupancy: region1 layer2; phi [rad]; strip", 
	   "h_", "(280,-3.141592654,3.141592654,192,0,384)", "strip:g_phi", rp1 && l2, "COLZ");
 
  draw_1D(targetDir_, "strip_dg_rm1_l1", ext_, treeDigis_, "Digi occupancy per strip number, region-1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", rm1 && l1);
  draw_1D(targetDir_, "strip_dg_rm1_l2", ext_, treeDigis_, "Digi occupancy per strip number, region-1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", rm1 && l2);
  draw_1D(targetDir_, "strip_dg_rp1_l1", ext_, treeDigis_, "Digi occupancy per strip number, region1 layer1;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", rp1 && l1);
  draw_1D(targetDir_, "strip_dg_rp1_l2", ext_, treeDigis_, "Digi occupancy per strip number, region1 layer2;strip number;entries", 
	  "h_", "(384,0.5,384.5)", "strip", rp1 && l2);
  
  /// Bunch crossing plots
  draw_bx(targetDir_, "strip_digi_bx_rm1_l1", ext_, treeDigis_, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1 && l1);
  draw_bx(targetDir_, "strip_digi_bx_rm1_l2", ext_, treeDigis_, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1 && l2);
  draw_bx(targetDir_, "strip_digi_bx_rp1_l1", ext_, treeDigis_, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1 && l1);
  draw_bx(targetDir_, "strip_digi_bx_rp1_l2", ext_, treeDigis_, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1 && l2);


  TTree * treePads_( (TTree *) dirAna_->Get( pads_ ) );
  if ( ! treePads_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    pads '" << pads_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  /// occupancy plots
  draw_occ(targetDir_, "pad_dg_xy_rm1_l1", ext_, treePads_, "Pad occupancy: region-1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1 && l1, "COLZ");
  draw_occ(targetDir_, "pad_dg_xy_rm1_l2", ext_, treePads_, "Pad occupancy: region-1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1 && l2, "COLZ");
  draw_occ(targetDir_, "pad_dg_xy_rp1_l1", ext_, treePads_, "Pad occupancy: region1, layer1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1 && l1, "COLZ");
  draw_occ(targetDir_, "pad_dg_xy_rp1_l2", ext_, treePads_, "Pad occupancy: region1, layer2; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1 && l2, "COLZ");

  draw_occ(targetDir_, "pad_dg_zr_rm1", ext_, treePads_, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ");
  draw_occ(targetDir_, "pad_dg_zr_rp1", ext_, treePads_, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ");

  draw_occ(targetDir_, "pad_dg_phipad_rm1_l1", ext_, treePads_, "Pad occupancy: region-1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654," + boost::lexical_cast< std::string >((double) (npads_/2.)) + ",0," + npadss + ")", "pad:g_phi", rm1 && l1, "COLZ");
  draw_occ(targetDir_, "pad_dg_phipad_rm1_l2", ext_, treePads_, "Pad occupancy: region-1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654," + boost::lexical_cast< std::string >((double) (npads_/2.)) + ",0," + npadss + ")", "pad:g_phi", rm1 && l2, "COLZ");
  draw_occ(targetDir_, "pad_dg_phipad_rp1_l1", ext_, treePads_, "Pad occupancy: region1 layer1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654," + boost::lexical_cast< std::string >((double) (npads_/2.)) + ",0," + npadss + ")", "pad:g_phi", rp1 && l1, "COLZ");
  draw_occ(targetDir_, "pad_dg_phipad_rp1_l2", ext_, treePads_, "Pad occupancy: region1 layer2; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654," + boost::lexical_cast< std::string >((double) (npads_/2.)) + ",0," + npadss + ")", "pad:g_phi", rp1 && l2, "COLZ");
 
  draw_1D(targetDir_, "pad_dg_rm1_l1", ext_, treePads_, "Digi occupancy per pad number, region-1 layer1;pad number;entries", 
	  "h_", "(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")", "pad", rm1 && l1);
  draw_1D(targetDir_, "pad_dg_rm1_l2", ext_, treePads_, "Digi occupancy per pad number, region-1 layer2;pad number;entries", 
	  "h_", "(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")", "pad", rm1 && l2);
  draw_1D(targetDir_, "pad_dg_rp1_l1", ext_, treePads_, "Digi occupancy per pad number, region1 layer1;pad number;entries", 
	  "h_", "(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")", "pad", rp1 && l1);
  draw_1D(targetDir_, "pad_dg_rp1_l2", ext_, treePads_, "Digi occupancy per pad number, region1 layer2;pad number;entries", 
	  "h_", "(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")", "pad", rp1 && l2);

  /// Bunch crossing plots
  draw_bx(targetDir_, "pad_dg_bx_rm1_l1", ext_, treePads_, "Bunch crossing: region-1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1 && l1);
  draw_bx(targetDir_, "pad_dg_bx_rm1_l2", ext_, treePads_, "Bunch crossing: region-1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1 && l2);
  draw_bx(targetDir_, "pad_dg_bx_rp1_l1", ext_, treePads_, "Bunch crossing: region1, layer1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1 && l1);
  draw_bx(targetDir_, "pad_dg_bx_rp1_l2", ext_, treePads_, "Bunch crossing: region1, layer2;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1 && l2);


  TTree * treeCoPads_( (TTree *) dirAna_->Get( copads_ ) );
  if ( ! treePads_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    copad '" << copads_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  /// occupancy plots
  draw_occ(targetDir_, "copad_dg_xy_rm1_l1", ext_, treeCoPads_, "Pad occupancy: region-1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rm1, "COLZ");
  draw_occ(targetDir_, "copad_dg_xy_rp1_l1", ext_, treeCoPads_, "Pad occupancy: region1; globalX [cm]; globalY [cm]", 
	   "h_", "(260,-260,260,260,-260,260)", "g_x:g_y", rp1, "COLZ");

  draw_occ(targetDir_, "copad_dg_zr_rm1", ext_, treeCoPads_, "Pad occupancy: region-1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,-573,-564,55,130,240)", "g_r:g_z", rm1, "COLZ");
  draw_occ(targetDir_, "copad_dg_zr_rp1", ext_, treeCoPads_, "Pad occupancy: region1; globalZ [cm]; globalR [cm]", 
	   "h_", "(200,564,573,55,130,240)", "g_r:g_z", rp1, "COLZ");

  draw_occ(targetDir_, "copad_dg_phipad_rm1_l1", ext_, treeCoPads_, "Pad occupancy: region-1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654," + boost::lexical_cast< std::string >((double) (npads_/2.)) + ",0," + npadss + ")", "pad:g_phi", rm1, "COLZ");
  draw_occ(targetDir_, "copad_dg_phipad_rp1_l1", ext_, treeCoPads_, "Pad occupancy: region1; phi [rad]; pad", 
	   "h_", "(280,-3.141592654,3.141592654," + boost::lexical_cast< std::string >((double) (npads_/2.)) + ",0," + npadss + ")", "pad:g_phi", rp1, "COLZ");
 
  draw_1D(targetDir_, "copad_dg_rm1_l1", ext_, treeCoPads_, "Digi occupancy per pad number, region-1;pad number;entries", 
	  "h_", "(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")", "pad", rm1);
  draw_1D(targetDir_, "copad_dg_rp1_l1", ext_, treeCoPads_, "Digi occupancy per pad number, region1;pad number;entries", 
	  "h_", "(" + npadss + ",0.5," + boost::lexical_cast< std::string >( (double) (npads_ + 0.5)) +  ")", "pad", rp1);

  /// Bunch crossing plots
  draw_bx(targetDir_, "copad_dg_bx_rm1", ext_, treeCoPads_, "Bunch crossing: region-1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rm1);
  draw_bx(targetDir_, "copad_dg_bx_rp1", ext_, treeCoPads_, "Bunch crossing: region1;bunch crossing;entries", 
	  "h_", "(11,-5.5,5.5)", "bx", rp1);

  /// Tracks
  TTree * treeTracks_( (TTree *) dirAna_->Get( tracks_ ) );
  if ( ! treeTracks_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    tracks '" << tracks_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  const TCut ok_eta("TMath::Abs(eta) > 1.64 && TMath::Abs(eta) < 2.12");
  const TCut ok_gL1sh("gem_sh_layer1 > 0");
  const TCut ok_gL2sh("gem_sh_layer2 > 0");
  const TCut ok_gL1dg("gem_dg_layer1 > 0");
  const TCut ok_gL2dg("gem_dg_layer2 > 0");
  const TCut ok_gL1pad("gem_pad_layer1 > 0");
  const TCut ok_gL2pad("gem_pad_layer2 > 0");

  /// digis
  draw_geff(targetDir_, "eff_eta_track_dg_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_dg_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL2dg, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_dg_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL2dg || ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_dg_gem_l1and2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL2dg && ok_gL1dg, "P", kBlue);

  draw_geff(targetDir_, "eff_phi_track_dg_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_dg_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2dg, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_dg_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2dg || ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_dg_gem_l1and2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2dg && ok_gL1dg, "P", kBlue);

  // digis with matched simhits
  draw_geff(targetDir_, "eff_eta_track_dg_sh_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_dg_sh_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2dg, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_dg_sh_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh || ok_gL2sh, ok_gL2dg || ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_dg_sh_gem_l1and2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh && ok_gL2sh, ok_gL2dg && ok_gL1dg, "P", kBlue);

  draw_geff(targetDir_, "eff_phi_track_dg_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && ok_gL1sh, ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_dg_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && ok_gL2sh, ok_gL2dg, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_dg_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && (ok_gL1sh || ok_gL2sh), ok_gL2dg || ok_gL1dg, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_dg_gem_l1and2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Digi in l1 and l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && (ok_gL1sh && ok_gL2sh), ok_gL2dg && ok_gL1dg, "P", kBlue);

  /// pads
  draw_geff(targetDir_, "eff_eta_track_pad_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL1pad, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_pad_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL2pad, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_pad_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL2pad || ok_gL1pad, "P", kBlue);

  draw_geff(targetDir_, "eff_phi_track_pad_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1pad, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_pad_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2pad, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_pad_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL2pad || ok_gL1pad, "P", kBlue);

  // pads with matched simhits
  draw_geff(targetDir_, "eff_eta_track_pad_sh_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh, ok_gL1pad, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_pad_sh_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL2sh, ok_gL2pad, "P", kBlue);
  draw_geff(targetDir_, "eff_eta_track_pad_sh_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh || ok_gL2sh, ok_gL2pad || ok_gL1pad, "P", kBlue);

  draw_geff(targetDir_, "eff_phi_track_pad_sh_gem_l1", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && ok_gL1sh, ok_gL1pad, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_pad_sh_gem_l2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && ok_gL2sh, ok_gL2pad, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_pad_sh_gem_l1or2", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM Pad in l1 or l2 with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && (ok_gL1sh || ok_gL2sh), ok_gL2pad || ok_gL1pad, "P", kBlue);

  /// copads
  draw_geff(targetDir_, "eff_eta_track_copad_gem", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", "", ok_gL1pad && ok_gL2pad, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_copad_gem", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM CoPad;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta, ok_gL1pad && ok_gL2pad, "P", kBlue);

  // copads with matched simhits
  draw_geff(targetDir_, "eff_eta_track_copad_sh_gem", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack |#eta|;Eff.", 
	    "h_", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_gL1sh && ok_gL2sh, ok_gL1pad && ok_gL2pad, "P", kBlue);
  draw_geff(targetDir_, "eff_phi_track_copad_sh_gem", ext_, treeTracks_, 
	    "Eff. for a SimTrack to have an associated GEM CoPad with a matched SimHit;SimTrack #phi [rad];Eff.", 
	    "h_", "(100,-3.141592654,3.141592654)", "phi", ok_eta && (ok_gL1sh && ok_gL2sh), ok_gL1pad && ok_gL2pad, "P", kBlue);

  // track plots
  draw_1D(targetDir_, "track_pt", ext_, treeTracks_, "Track p_{T};Track p_{T} [GeV];Entries", "h_", "(100,0,200)", "pt", "");
  draw_1D(targetDir_, "track_eta", ext_, treeTracks_, "Track |#eta|;Track |#eta|;Entries", "h_", "(100,1.5,2.2)", "eta", "");
  draw_1D(targetDir_, "track_phi", ext_, treeTracks_, "Track #phi;Track #phi [rad];Entries", "h_", "(100,-3.141592654,3.141592654)", "phi", "");

  return returnStatus_;
}

void draw_geff(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	       TString to_draw, TCut denom_cut, TCut extra_num_cut, TString opt, int color, int marker_st, float marker_sz)
{
  TCanvas* c( new TCanvas("c","c",600,600) );
  c->Clear();
  gPad->SetGrid(1);    
  gStyle->SetStatStyle(0);
  gStyle->SetOptStat(1110);

  t->Draw(to_draw + ">>num_" + h_name + h_bins, denom_cut && extra_num_cut, "goff");
  TH1F* num((TH1F*) gDirectory->Get("num_" + h_name)->Clone("eff_" + h_name));
  
  t->Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff");
  TH1F* den((TH1F*) gDirectory->Get("denom_" + h_name)->Clone("denom_" + h_name));
  
  TGraphAsymmErrors *eff( new TGraphAsymmErrors(num, den));
  
  if (!opt.Contains("same")) {
    num->Reset();
    num->GetYaxis()->SetRangeUser(0.,1.05);
    num->SetStats(0);
    num->SetTitle(title);
    num->Draw();
  }
  eff->SetLineWidth(2);
  eff->SetLineColor(color);
  eff->SetMarkerStyle(marker_st);
  eff->SetMarkerColor(color);
  eff->SetMarkerSize(marker_sz);
  eff->Draw(opt + " same");

  // Do fit in the flat region
  bool etaPlot(c_title.Contains("eta"));
  const double xmin(etaPlot ? 1.64 : -999.);
  const double xmax(etaPlot ? 2.12 : 999.);
  TF1 *f1 = new TF1("fit1","pol0", xmin, xmax);
  TFitResultPtr r = eff->Fit("fit1","RQS");
  TPaveStats *ptstats = new TPaveStats(0.25,0.35,0.75,0.55,"brNDC");
  ptstats->SetName("stats");
  ptstats->SetBorderSize(0);
  ptstats->SetLineWidth(0);
  ptstats->SetFillColor(0);
  ptstats->SetTextAlign(11);
  ptstats->SetTextFont(42);
  ptstats->SetTextSize(.05);
  ptstats->SetTextColor(kRed);
  ptstats->SetOptStat(0);
  ptstats->SetOptFit(1111);

  std::stringstream sstream;
  sstream << TMath::Nint(r->Chi2());
  const TString chi2(boost::lexical_cast< std::string >(sstream.str()));
  sstream.str(std::string());
  sstream << r->Ndf();
  const TString ndf(boost::lexical_cast< std::string >(sstream.str()));
  sstream.str(std::string());
  sstream << TMath::Nint(r->Prob()*100);
  const TString prob(boost::lexical_cast< std::string >(sstream.str()));
  sstream.str(std::string());
  sstream << std::setprecision(4) << f1->GetParameter(0) * 100;
  const TString p0(boost::lexical_cast< std::string >(sstream.str()));
  sstream.str(std::string());
  sstream << std::setprecision(2) << f1->GetParError(0) * 100;
  const TString p0e(boost::lexical_cast< std::string >(sstream.str()));
  ptstats->AddText("#chi^{2} / ndf: " + chi2 + "/" + ndf);
//   ptstats->AddText("Fit probability: " + prob + " %");
//   ptstats->AddText("Fitted efficiency = " + p0 + " #pm " + p0e + " %" );
  ptstats->AddText("Fitted efficiency: " + p0 + " #pm " + p0e + " %");
  ptstats->Draw("same");

  TPaveText *pt = new TPaveText(0.09899329,0.9178322,0.8993289,0.9737762,"blNDC");
  pt->SetName("title");
  pt->SetBorderSize(1);
  pt->SetFillColor(0);
  pt->SetFillStyle(0);
  pt->SetTextFont(42);
  pt->AddText(eff->GetTitle());
  pt->Draw("same");
  c->Modified();
  c->SaveAs(target_dir + c_title + ext);
  delete num;
  delete den;
  delete eff;
  delete c;
}

  
void draw_occ(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	      TString to_draw, TCut cut, TString opt)
{
  gStyle->SetStatStyle(0);
  gStyle->SetOptStat(1110);
  TCanvas* c = new TCanvas("c","c",600,600);
  t->Draw(to_draw + ">>" + h_name + h_bins, cut); 
  TH2F* h = (TH2F*) gDirectory->Get(h_name)->Clone(h_name);
  h->SetTitle(title);
  h->SetLineWidth(2);
  h->SetLineColor(kBlue);
  h->Draw(opt);
  c->SaveAs(target_dir + c_title + ext);
  delete h;
  delete c;
}

void draw_1D(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	      TString to_draw, TCut cut, TString opt)
{
  gStyle->SetStatStyle(0);
  gStyle->SetOptStat(1110);
  TCanvas* c = new TCanvas("c","c",600,600);
  t->Draw(to_draw + ">>" + h_name + h_bins, cut); 
  TH1F* h = (TH1F*) gDirectory->Get(h_name)->Clone(h_name);
  h->SetTitle(title);
  h->SetLineWidth(2);
  h->SetLineColor(kBlue);
  h->Draw(opt);
  h->SetMinimum(0.);
  c->SaveAs(target_dir + c_title + ext);
  delete h;
  delete c;
}

void draw_bx(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
	      TString to_draw, TCut cut, TString opt)
{
  gStyle->SetStatStyle(0);
  gStyle->SetOptStat(1110);
  TCanvas* c = new TCanvas("c","c",600,600);
  t->Draw(to_draw + ">>" + h_name + h_bins, cut); 
  gPad->SetLogy();
  TH1F* h = (TH1F*) gDirectory->Get(h_name)->Clone(h_name);
  h->SetTitle(title);
  h->SetLineWidth(2);
  h->SetLineColor(kBlue);
  h->Draw(opt);
  h->SetMinimum(1.);
  c->SaveAs(target_dir + c_title + ext);
  delete h;
  delete c;
}
