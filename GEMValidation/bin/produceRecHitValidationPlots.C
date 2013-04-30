
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

using namespace std;

// void draw_geff(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
// 	       TString to_draw, TCut denom_cut, TCut extra_num_cut, TString opt = "", int color = kBlue, int marker_st = 1, float marker_sz = 1.);
void draw_occ(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
 	      TString to_draw, TCut cut, TString opt = "");
void draw_1D(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
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
  const int nregion_( process_.getParameter< int >( "nregion" ) );
  const int npart_( process_.getParameter< int >( "npart" ) );
   
  // Constants 
  const TString analyzer_( "GEMRecHitAnalyzer" );
  const TString rechits_( "GEMRecHitTree" ); 
  const TString simTracks_( "Tracks" ); 
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
              << "    rechits '" << analyzer_ << "' does not exist in input file" << std::endl;
    returnStatus_ += 0x20;
    return returnStatus_;
  }

  TTree * treeHits_( (TTree *) dirAna_->Get( rechits_ ) );
  if ( ! treeHits_ ) {
    std::cout << argv[ 0 ] << " --> WARNING:" << std::endl
              << "    rechits '" << rechits_ << "' does not exist in directory" << std::endl;
    returnStatus_ += 0x30;
    return returnStatus_;
  }

  /// occupancy
  draw_occ(targetDir_, "localrh_xy_rm1_l1", ext_, treeHits_, " SimHit occupancy: region-1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", rm1 && l1, "COLZ");
  draw_occ(targetDir_, "localrh_xy_rm1_l2", ext_, treeHits_, " SimHit occupancy: region-1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", rm1 && l2, "COLZ");
  draw_occ(targetDir_, "localrh_xy_rp1_l1", ext_, treeHits_, " SimHit occupancy: region1, layer1;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", rp1 && l1, "COLZ");
  draw_occ(targetDir_, "localrh_xy_rp1_l2", ext_, treeHits_, " SimHit occupancy: region1, layer2;globalX [cm];globalY [cm]", 
	   "h_", "(100,-260,260,100,-260,260)", "globalY:globalX", rp1 && l2, "COLZ");
  
  draw_occ(targetDir_, "localrh_zr_rm1", ext_, treeHits_, " SimHit occupancy: region-1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,-573,-564,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rm1, "COLZ");
  draw_occ(targetDir_, "localrh_zr_rp1", ext_, treeHits_, " SimHit occupancy: region1;globalZ [cm];globalR [cm]", 
	   "h_", "(200,564,573,110,130,240)", "sqrt(globalX*globalX+globalY*globalY):globalZ", rp1, "COLZ");
  
  /// eta occupancy plot
  int region=0;
  int layer=0;
  int roll=0;  
  TBranch *b_region;
  TBranch *b_layer;
  TBranch *b_roll;
  treeHits_->SetBranchAddress("region", &region, &b_region);
  treeHits_->SetBranchAddress("layer", &layer, &b_layer);
  treeHits_->SetBranchAddress("roll", &roll, &b_roll);
  TH1D* h = new TH1D("h", " SimHit occupancy in eta partitions; occupancy in #eta partition; entries",4*npart_,1.,1.+4*npart_);
  int nbytes = 0;
  int nb = 0;
  for (Long64_t jentry=0; jentry<treeHits_->GetEntriesFast();jentry++) {
    Long64_t ientry = treeHits_->LoadTree(jentry);
    if (ientry < 0) break;
    nb = treeHits_->GetEntry(jentry);   
    nbytes += nb;
    h->Fill(roll + (layer==2? npart_:0) + (region==1? 2.*npart_:0 ) );
  }    
  TCanvas* c = new TCanvas("c","c",600,600);
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
  h->SetLineWidth(2);
  h->SetLineColor(kBlue);
  h->Draw("");        
  c->SaveAs(targetDir_ +"localrh_globalEta" + ext_);
  
  return returnStatus_;
}

// void draw_geff(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
// 	       TString to_draw, TCut denom_cut, TCut extra_num_cut, TString opt, int color, int marker_st, float marker_sz)
// {
//   TCanvas* c( new TCanvas("c","c",600,600) );
//   c->Clear();
//   gPad->SetGrid(1);    

//   t->Draw(to_draw + ">>num_" + h_name + h_bins, denom_cut && extra_num_cut, "goff");
//   TH1F* num((TH1F*) gDirectory->Get("num_" + h_name)->Clone("eff_" + h_name));
  
//   t->Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff");
//   TH1F* den((TH1F*) gDirectory->Get("denom_" + h_name)->Clone("denom_" + h_name));
  
//   TGraphAsymmErrors *eff( new TGraphAsymmErrors(num, den));
  
//   if (!opt.Contains("same")) {
//     num->Reset();
//     num->GetYaxis()->SetRangeUser(0.,1.05);
//     num->SetStats(0);
//     num->SetTitle(title);
//     num->Draw();
//   }
//   eff->SetLineWidth(2);
//   eff->SetLineColor(color);
//   eff->SetMarkerStyle(marker_st);
//   eff->SetMarkerColor(color);
//   eff->SetMarkerSize(marker_sz);
//   eff->Draw(opt + " same");

//   // Do fit in the flat region
//   bool etaPlot(c_title.Contains("eta"));
//   const double xmin(etaPlot ? 1.64 : -999.);
//   const double xmax(etaPlot ? 2.12 : 999.);
//   TF1 *f1 = new TF1("fit1","pol0", xmin, xmax);
//   TFitResultPtr r = eff->Fit("fit1","RQS");
//   TPaveStats *ptstats = new TPaveStats(0.25,0.35,0.75,0.55,"brNDC");
//   ptstats->SetName("stats");
//   ptstats->SetBorderSize(0);
//   ptstats->SetLineWidth(0);
//   ptstats->SetFillColor(0);
//   ptstats->SetTextAlign(11);
//   ptstats->SetTextFont(42);
//   ptstats->SetTextSize(.05);
//   ptstats->SetTextColor(kRed);
//   ptstats->SetOptStat(0);
//   ptstats->SetOptFit(1111);

//   std::stringstream sstream;
//   sstream << TMath::Nint(r->Chi2());
//   const TString chi2(boost::lexical_cast< std::string >(sstream.str()));
//   sstream.str(std::string());
//   sstream << r->Ndf();
//   const TString ndf(boost::lexical_cast< std::string >(sstream.str()));
//   sstream.str(std::string());
//   sstream << TMath::Nint(r->Prob()*100);
//   const TString prob(boost::lexical_cast< std::string >(sstream.str()));
//   sstream.str(std::string());
//   sstream << f1->GetParameter(0);//<< std::setprecision(4) 
//   const TString p0(boost::lexical_cast< std::string >(sstream.str()));
//   sstream.str(std::string());
//   sstream << std::setprecision(2) << f1->GetParError(0);
//   const TString p0e(boost::lexical_cast< std::string >(sstream.str()));
//   ptstats->AddText("#chi^{2} / ndf: " + chi2 + "/" + ndf);
// //   ptstats->AddText("Fit probability: " + prob + " %");
// //   ptstats->AddText("Fitted efficiency = " + p0 + " #pm " + p0e + " %" );
//   ptstats->AddText("Fitted efficiency: " + p0 + " %");
//   ptstats->Draw("same");
//   c->Modified();
//   c->SaveAs(target_dir + c_title + ext);
//   delete num;
//   delete den;
//   delete eff;
//   delete c;
// }

void draw_occ(TString target_dir, TString c_title, TString ext, TTree *t, TString title, TString h_name, TString h_bins,
 	      TString to_draw, TCut cut, TString opt)
{
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
 
