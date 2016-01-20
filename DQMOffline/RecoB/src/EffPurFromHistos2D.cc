#include "DQMOffline/RecoB/interface/EffPurFromHistos2D.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <iostream>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 

using namespace std;
using namespace RecoBTag;

EffPurFromHistos2D::EffPurFromHistos2D ( const std::string & ext, TH2F * h_d, TH2F * h_u,
                                     TH2F * h_s, TH2F * h_c, TH2F * h_b, TH2F * h_g, TH2F * h_ni,
                                     TH2F * h_dus, TH2F * h_dusg, TH2F * h_pu,
				     const std::string& label, const unsigned int& mc, 
				     int nBinX, double startOX, double endOX, int nBinY, double startOY, double endOY) :
        fromDiscriminatorDistr(false),
	histoExtension(ext), effVersusDiscr_d(h_d), effVersusDiscr_u(h_u),
        effVersusDiscr_s(h_s), effVersusDiscr_c(h_c), effVersusDiscr_b(h_b),
        effVersusDiscr_g(h_g), effVersusDiscr_ni(h_ni), effVersusDiscr_dus(h_dus),
        effVersusDiscr_dusg(h_dusg), effVersusDiscr_pu(h_pu),
	nBinOutputX(nBinX), startOutputX(startOX), endOutputX(endOX), nBinOutputY(nBinY), startOutputY(startOY), endOutputY(endOY),  
        mcPlots_(mc), doCTagPlots_(false), label_(label)
{
  // consistency check
  check();
}

EffPurFromHistos2D::EffPurFromHistos2D (const FlavourHistograms2D<double, double> * dDiscriminatorFC, 
				    const std::string& label, const unsigned int& mc, 
                                    DQMStore::IBooker & ibook, 
                                    int nBinX, double startOX, double endOX, int nBinY, double startOY, double endOY) :
	  fromDiscriminatorDistr(true), nBinOutputX(nBinX), startOutputX(startOX), endOutputX(endOX), nBinOutputY(nBinY), startOutputY(startOY), endOutputY(endOY), mcPlots_(mc), doCTagPlots_(false), label_(label){
  histoExtension = "_"+dDiscriminatorFC->baseNameTitle();

  discrNoCutEffic = new FlavourHistograms2D<double, double> (
        "totalEntries" + histoExtension, "Total Entries: " + dDiscriminatorFC->baseNameDescription(),
        dDiscriminatorFC->nBinsX(), dDiscriminatorFC->lowerBoundX(), dDiscriminatorFC->upperBoundX(), dDiscriminatorFC->nBinsY(), 
        dDiscriminatorFC->lowerBoundY(), dDiscriminatorFC->upperBoundY(), 
        false, label, mcPlots_, false, ibook);
  // conditional discriminator cut for efficiency histos

  discrCutEfficScan = new FlavourHistograms2D<double, double> (
	"effVsDiscrCut" + histoExtension, "Eff. vs Disc. Cut: " + dDiscriminatorFC->baseNameDescription(),
	dDiscriminatorFC->nBinsX(), dDiscriminatorFC->lowerBoundX(), dDiscriminatorFC->upperBoundX(), dDiscriminatorFC->nBinsY(),
        dDiscriminatorFC->lowerBoundY(), dDiscriminatorFC->upperBoundY(),
        false, label, mcPlots_, false, ibook);
  discrCutEfficScan->SetMinimum(1E-4);

  if (mcPlots_){

    if(mcPlots_>2){
      effVersusDiscr_d =    discrCutEfficScan->histo_d   ();
      effVersusDiscr_u =    discrCutEfficScan->histo_u   ();
      effVersusDiscr_s =    discrCutEfficScan->histo_s   ();
      effVersusDiscr_g =    discrCutEfficScan->histo_g   ();
      effVersusDiscr_dus =  discrCutEfficScan->histo_dus ();
    }
    else{
      effVersusDiscr_d   =  0;
      effVersusDiscr_u   =  0;
      effVersusDiscr_s   =  0;
      effVersusDiscr_g   =  0;
      effVersusDiscr_dus =  0;
    }
    effVersusDiscr_c =    discrCutEfficScan->histo_c   ();
    effVersusDiscr_b =    discrCutEfficScan->histo_b   ();
    effVersusDiscr_ni =   discrCutEfficScan->histo_ni  ();
    effVersusDiscr_dusg = discrCutEfficScan->histo_dusg();
    effVersusDiscr_pu = discrCutEfficScan->histo_pu();


    if(mcPlots_>2){
      effVersusDiscr_d->SetXTitle ( "Discriminant" );
      effVersusDiscr_d->GetXaxis()->SetTitleOffset ( 0.75 );
      effVersusDiscr_u->SetXTitle ( "Discriminant" );
      effVersusDiscr_u->GetXaxis()->SetTitleOffset ( 0.75 );
      effVersusDiscr_s->SetXTitle ( "Discriminant" );
      effVersusDiscr_s->GetXaxis()->SetTitleOffset ( 0.75 );
      effVersusDiscr_g->SetXTitle ( "Discriminant" );
      effVersusDiscr_g->GetXaxis()->SetTitleOffset ( 0.75 );
      effVersusDiscr_dus->SetXTitle ( "Discriminant" );
      effVersusDiscr_dus->GetXaxis()->SetTitleOffset ( 0.75 );
    }
    effVersusDiscr_c->SetXTitle ( "Discriminant" );
    effVersusDiscr_c->GetXaxis()->SetTitleOffset ( 0.75 );
    effVersusDiscr_b->SetXTitle ( "Discriminant" );
    effVersusDiscr_b->GetXaxis()->SetTitleOffset ( 0.75 );
    effVersusDiscr_ni->SetXTitle ( "Discriminant" );
    effVersusDiscr_ni->GetXaxis()->SetTitleOffset ( 0.75 );
    effVersusDiscr_dusg->SetXTitle ( "Discriminant" );
    effVersusDiscr_dusg->GetXaxis()->SetTitleOffset ( 0.75 );
    effVersusDiscr_pu->SetXTitle ( "Discriminant" );
    effVersusDiscr_pu->GetXaxis()->SetTitleOffset ( 0.75 );
  }
  else{
    effVersusDiscr_d =    0;
    effVersusDiscr_u =    0;
    effVersusDiscr_s =    0;
    effVersusDiscr_c =    0;
    effVersusDiscr_b =    0;
    effVersusDiscr_g =    0;
    effVersusDiscr_ni =   0;
    effVersusDiscr_dus =  0;
    effVersusDiscr_dusg = 0;
    effVersusDiscr_pu = 0;
  }

  // discr. for computation
  vector<TH2F*> discrCfHistos = dDiscriminatorFC->getHistoVector();
  // discr no cut
  vector<TH2F*> discrNoCutHistos = discrNoCutEffic->getHistoVector();
  // discr no cut
  vector<TH2F*> discrCutHistos = discrCutEfficScan->getHistoVector();

  const int& dimHistos = discrCfHistos.size(); // they all have the same size

  // DISCR-CUT LOOP:
  // fill the histos for eff-pur computations by scanning the discriminatorFC histogram

  // better to loop over bins -> discrCut no longer needed
  const int& nBinsX = dDiscriminatorFC->nBinsX();
  const int& nBinsY = dDiscriminatorFC->nBinsY();

  // loop over flavours
  for ( int iFlav = 0; iFlav < dimHistos; iFlav++ ) {
    if (discrCfHistos[iFlav] == 0) continue;
    discrNoCutHistos[iFlav]->SetXTitle ( "Discriminant" );
    discrNoCutHistos[iFlav]->GetXaxis()->SetTitleOffset ( 0.75 );

    // In Root histos, bin counting starts at 1 to nBins.
    // bin 0 is the underflow, and nBins+1 is the overflow.
    const double& nJetsFlav = discrCfHistos[iFlav]->GetEntries ();
    //double sumX = discrCfHistos[iFlav]->GetBinContent( nBinsX+1 ); //+1 to get the overflow.
    double sum = discrCfHistos[iFlav]->GetBinContent( nBinsX+1,nBinsY+1 );    

    for ( int iDiscrX = nBinsX; iDiscrX > 0 ; --iDiscrX ) {
	for ( int iDiscrY = nBinsY; iDiscrY > 0 ; --iDiscrY ) {
      // fill all jets into NoCut histo
      discrNoCutHistos[iFlav]->SetBinContent ( iDiscrX, iDiscrY, nJetsFlav );
      discrNoCutHistos[iFlav]->SetBinError   ( iDiscrX, iDiscrY, sqrt(nJetsFlav) );
      sum += discrCfHistos[iFlav]->GetBinContent( iDiscrX, iDiscrY );
      discrCutHistos[iFlav]->SetBinContent ( iDiscrX, iDiscrY, sum );
      discrCutHistos[iFlav]->SetBinError   ( iDiscrX, iDiscrY, sqrt(sum) );
	}
    }
  }

  // divide to get efficiency vs. discriminator cut from absolute numbers
  discrCutEfficScan->divide ( *discrNoCutEffic );  // does: histos including discriminator cut / flat histo
  //discrCutEfficScan->setEfficiencyFlag();
}


EffPurFromHistos2D::~EffPurFromHistos2D () {
  /*  delete EffFlavVsBEff_d   ;
  delete EffFlavVsBEff_u   ;
  delete EffFlavVsBEff_s   ;
  delete EffFlavVsBEff_c   ;
  delete EffFlavVsBEff_b   ;
  delete EffFlavVsBEff_g   ;
  delete EffFlavVsBEff_ni  ;
  delete EffFlavVsBEff_dus ;
  delete EffFlavVsBEff_dusg;
  if ( fromDiscriminatorDistr) {
    delete discrNoCutEffic;
    delete discrCutEfficScan;
    }*/
}




void EffPurFromHistos2D::epsPlot(const std::string & name)
{
  if ( fromDiscriminatorDistr) {
    discrNoCutEffic->epsPlot(name);
    discrCutEfficScan->epsPlot(name);
  }
  plot(name, ".eps");
}

void EffPurFromHistos2D::psPlot(const std::string & name)
{
  plot(name, ".ps");
}

void EffPurFromHistos2D::plot(const std::string & name, const std::string & ext)
{
   std::string hX = "";
	 std::string Title = "";
   if(!doCTagPlots_){
	   hX = "FlavEffVsBEff";
		 Title = "b";
   }
   else{
     hX = "FlavEffVsCEff";
		 Title = "c";
   }
   TCanvas tc ((hX +histoExtension).c_str() ,
	("Flavour misidentification vs. " + Title + "-tagging efficiency " + histoExtension).c_str());
   plot(&tc);
   tc.Print((name + hX + histoExtension + ext).c_str());
}

void EffPurFromHistos2D::plot (TPad * plotCanvas /* = 0 */) {

//fixme:
/*
  bool btppNI = false;
  bool btppColour = true;
*/
//   if ( !btppTitle ) gStyle->SetOptTitle ( 0 );
  setTDRStyle()->cd();

  if (plotCanvas)
    plotCanvas->cd();
  
  gPad->UseCurrentStyle();
  gPad->SetFillColor ( 0 );
  gPad->SetLogy  ( 1 );
  gPad->SetGridx ( 1 );
  gPad->SetGridy ( 1 );
/*
  int col_c  ;
  int col_g  ;
  int col_dus;
  int col_ni ;

  int mStyle_c  ;
  int mStyle_g  ;
  int mStyle_dus;
  int mStyle_ni ;
*/
  // marker size (same for all)
  //float mSize = gPad->GetWh() * gPad->GetHNDC() / 500.; //1.2;
/*
  if ( btppColour ) {
    col_c    = 6;
    col_g    = 3; // g in green
    col_dus  = 4; // uds in blue
    col_ni   = 5; // ni in ??
    mStyle_c   = 20;
    mStyle_g   = 20;
    mStyle_dus = 20;
    mStyle_ni  = 20;
  }
  else {
    col_c    = 1;
    col_g    = 1;
    col_dus  = 1;
    col_ni   = 1;
    mStyle_c   = 22;
    mStyle_g   = 29;
    mStyle_dus = 20;
    mStyle_ni  = 27;
  }
 
  TString Title = "";
  if(!doCTagPlots_){
        Title = "b";
  }
  else{
        Title = "c";
  } 

  // for the moment: plot c,dus,g
  if(mcPlots_>2){
    EffFlavVsXEff_dus ->getTH1F()->GetXaxis()->SetTitle ( Title + "-jet efficiency" );
    EffFlavVsXEff_dus ->getTH1F()->GetYaxis()->SetTitle ( "non " + Title + "-jet efficiency");
    EffFlavVsXEff_dus ->getTH1F()->GetYaxis()->SetTitleOffset ( 0.25 );
    EffFlavVsXEff_dus ->getTH1F()->SetMaximum     ( 1.1 );
    EffFlavVsXEff_dus ->getTH1F()->SetMinimum     ( 1.e-5 );
    EffFlavVsXEff_dus ->getTH1F()->SetMarkerColor ( col_dus );
    EffFlavVsXEff_dus ->getTH1F()->SetLineColor   ( col_dus );
    EffFlavVsXEff_dus ->getTH1F()->SetMarkerSize  ( mSize );
    EffFlavVsXEff_dus ->getTH1F()->SetMarkerStyle ( mStyle_dus );
    EffFlavVsXEff_dus ->getTH1F()->SetStats     ( false );
    EffFlavVsXEff_dus ->getTH1F()->Draw("pe");

    EffFlavVsXEff_g   ->getTH1F()->SetMarkerColor ( col_g );
    EffFlavVsXEff_g   ->getTH1F()->SetLineColor   ( col_g );
    EffFlavVsXEff_g   ->getTH1F()->SetMarkerSize  ( mSize );
    EffFlavVsXEff_g   ->getTH1F()->SetMarkerStyle ( mStyle_g );
    EffFlavVsXEff_g   ->getTH1F()->SetStats     ( false );
    EffFlavVsXEff_g   ->getTH1F()->Draw("peSame");
  }

  EffFlavVsXEff_c   ->getTH1F()->SetMarkerColor ( col_c );
  EffFlavVsXEff_c   ->getTH1F()->SetLineColor   ( col_c );
  EffFlavVsXEff_c   ->getTH1F()->SetMarkerSize  ( mSize );
  EffFlavVsXEff_c   ->getTH1F()->SetMarkerStyle ( mStyle_c );
  EffFlavVsXEff_c   ->getTH1F()->SetStats     ( false );
  EffFlavVsXEff_c   ->getTH1F()->Draw("peSame");

  if(mcPlots_>2){
    EffFlavVsXEff_d ->getTH1F()-> SetMinimum(0.01);
    EffFlavVsXEff_u ->getTH1F()-> SetMinimum(0.01);
    EffFlavVsXEff_s ->getTH1F()-> SetMinimum(0.01);
    EffFlavVsXEff_g ->getTH1F()-> SetMinimum(0.01);
    EffFlavVsXEff_dus ->getTH1F()-> SetMinimum(0.01);
  }
  EffFlavVsXEff_c ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsXEff_b ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsXEff_ni ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsXEff_dusg ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsXEff_pu ->getTH1F()-> SetMinimum(0.01);
*/
  DUSG_reject_vs_B_reject_at_cEff->getTH2F()-> SetMinimum(0.01);

  // plot separately u,d and s
//  EffFlavVsXEff_d ->GetXaxis()->SetTitle ( Title + "-jet efficiency" );
//  EffFlavVsXEff_d ->GetYaxis()->SetTitle ( "non " + Title + "-jet efficiency" );
//  EffFlavVsXEff_d ->GetYaxis()->SetTitleOffset ( 1.25 );
//  EffFlavVsXEff_d ->SetMaximum     ( 1.1 );
//  EffFlavVsXEff_d ->SetMinimum     ( 1.e-5 );
//  EffFlavVsXEff_d ->SetMarkerColor ( col_dus );
//  EffFlavVsXEff_d ->SetLineColor   ( col_dus );
//  EffFlavVsXEff_d ->SetMarkerSize  ( mSize );
//  EffFlavVsXEff_d ->SetMarkerStyle ( mStyle_dus );
//  EffFlavVsXEff_d ->SetStats     ( false );
//  EffFlavVsXEff_d ->Draw("pe");
//
//  EffFlavVsXEff_u   ->SetMarkerColor ( col_g );
//  EffFlavVsXEff_u   ->SetLineColor   ( col_g );
//  EffFlavVsXEff_u   ->SetMarkerSize  ( mSize );
//  EffFlavVsXEff_u   ->SetMarkerStyle ( mStyle_g );
//  EffFlavVsXEff_u   ->SetStats     ( false );
//  EffFlavVsXEff_u   ->Draw("peSame");
//
//  EffFlavVsXEff_s   ->SetMarkerColor ( col_c );
//  EffFlavVsXEff_s   ->SetLineColor   ( col_c );
//  EffFlavVsXEff_s   ->SetMarkerSize  ( mSize );
//  EffFlavVsXEff_s   ->SetMarkerStyle ( mStyle_c );
//  EffFlavVsXEff_s   ->SetStats     ( false );
//  EffFlavVsXEff_s   ->Draw("peSame");
/*
  // only if asked: NI
  if ( btppNI ) {
    EffFlavVsXEff_ni   ->getTH1F()->SetMarkerColor ( col_ni );
    EffFlavVsXEff_ni   ->getTH1F()->SetLineColor   ( col_ni );
    EffFlavVsXEff_ni   ->getTH1F()->SetMarkerSize  ( mSize );
    EffFlavVsXEff_ni   ->getTH1F()->SetMarkerStyle ( mStyle_ni );
    EffFlavVsXEff_ni   ->getTH1F()->SetStats     ( false );
    EffFlavVsXEff_ni   ->getTH1F()->Draw("peSame");
  }
*/
}


void EffPurFromHistos2D::check () {
  // number of bins
  int nBins_d    = 0;
  int nBins_u    = 0;
  int nBins_s    = 0;
  int nBins_g    = 0;
  int nBins_dus  = 0;
  if(mcPlots_>2){
    nBins_d    = effVersusDiscr_d    -> GetNbinsX();
    nBins_u    = effVersusDiscr_u    -> GetNbinsX();
    nBins_s    = effVersusDiscr_s    -> GetNbinsX();
    nBins_g    = effVersusDiscr_g    -> GetNbinsX();
    nBins_dus  = effVersusDiscr_dus  -> GetNbinsX();
  }
  const int& nBins_c    = effVersusDiscr_c    -> GetNbinsX();
  const int& nBins_b    = effVersusDiscr_b    -> GetNbinsX();
  const int& nBins_ni   = effVersusDiscr_ni   -> GetNbinsX();
  const int& nBins_dusg = effVersusDiscr_dusg -> GetNbinsX();
  const int& nBins_pu   = effVersusDiscr_pu   -> GetNbinsX();

  const bool& lNBins =
    ( (nBins_d == nBins_u    &&
       nBins_d == nBins_s    &&
       nBins_d == nBins_c    &&
       nBins_d == nBins_b    &&
       nBins_d == nBins_g    &&
       nBins_d == nBins_ni   &&
       nBins_d == nBins_dus  &&
       nBins_d == nBins_dusg)||
      (nBins_c == nBins_b    &&
       nBins_c == nBins_dusg &&
       nBins_c == nBins_ni   &&
       nBins_c == nBins_pu)    );

  if ( !lNBins ) {
    throw cms::Exception("Configuration")
      << "Input histograms do not all have the same number of bins!\n";
  }

  // start
  float sBin_d    = 0;
  float sBin_u    = 0;
  float sBin_s    = 0;
  float sBin_g    = 0;
  float sBin_dus  = 0;
  if(mcPlots_>2){
    sBin_d    = effVersusDiscr_d    -> GetBinCenter(1);
    sBin_u    = effVersusDiscr_u    -> GetBinCenter(1);
    sBin_s    = effVersusDiscr_s    -> GetBinCenter(1);
    sBin_g    = effVersusDiscr_g    -> GetBinCenter(1);
    sBin_dus  = effVersusDiscr_dus  -> GetBinCenter(1);
  }
  const float& sBin_c    = effVersusDiscr_c    -> GetBinCenter(1);
  const float& sBin_b    = effVersusDiscr_b    -> GetBinCenter(1);
  const float& sBin_ni   = effVersusDiscr_ni   -> GetBinCenter(1);
  const float& sBin_dusg = effVersusDiscr_dusg -> GetBinCenter(1);
  const float& sBin_pu   = effVersusDiscr_pu   -> GetBinCenter(1);

  const bool& lSBin =
    ( (sBin_d == sBin_u    &&
       sBin_d == sBin_s    &&
       sBin_d == sBin_c    &&
       sBin_d == sBin_b    &&
       sBin_d == sBin_g    &&
       sBin_d == sBin_ni   &&
       sBin_d == sBin_dus  &&
       sBin_d == sBin_dusg)||
      (sBin_c == sBin_b    &&
       sBin_c == sBin_dusg &&
       sBin_c == sBin_ni   &&
       sBin_c == sBin_pu)    );

  if ( !lSBin ) {
    throw cms::Exception("Configuration")
      << "EffPurFromHistos::check() : Input histograms do not all have the same start bin!\n";
  }
  
  // end
  float eBin_d    = 0;
  float eBin_u    = 0;
  float eBin_s    = 0;
  float eBin_g    = 0;
  float eBin_dus  = 0;
  if(mcPlots_>2){
    eBin_d    = effVersusDiscr_d    -> GetBinCenter( nBins_d - 1 );
    eBin_u    = effVersusDiscr_u    -> GetBinCenter( nBins_d - 1 );
    eBin_s    = effVersusDiscr_s    -> GetBinCenter( nBins_d - 1 );
    eBin_g    = effVersusDiscr_g    -> GetBinCenter( nBins_d - 1 );
    eBin_dus  = effVersusDiscr_dus  -> GetBinCenter( nBins_d - 1 );
  }
  const float& eBin_c    = effVersusDiscr_c    -> GetBinCenter( nBins_d - 1 );
  const float& eBin_b    = effVersusDiscr_b    -> GetBinCenter( nBins_d - 1 );
  const float& eBin_ni   = effVersusDiscr_ni   -> GetBinCenter( nBins_d - 1 );
  const float& eBin_dusg = effVersusDiscr_dusg -> GetBinCenter( nBins_d - 1 );
  const float& eBin_pu   = effVersusDiscr_pu   -> GetBinCenter( nBins_d - 1 );

  const bool& lEBin =
    ( (eBin_d == eBin_u    &&
       eBin_d == eBin_s    &&
       eBin_d == eBin_c    &&
       eBin_d == eBin_b    &&
       eBin_d == eBin_g    &&
       eBin_d == eBin_ni   &&
       eBin_d == eBin_dus  &&
       eBin_d == eBin_dusg)||
      (eBin_c == eBin_b    &&
       eBin_c == eBin_dusg &&
       eBin_c == eBin_ni   &&
       eBin_c == eBin_pu)     );

  if ( !lEBin ) {
    throw cms::Exception("Configuration")
      << "EffPurFromHistos::check() : Input histograms do not all have the same end bin!\n";
  } 
}

void EffPurFromHistos2D::compute (DQMStore::IBooker & ibook)
{

  if (!mcPlots_) {
    DUSG_reject_vs_B_reject_at_cEff =0;
    return; 
 
  }

  // to have shorter names ......
  const std::string & hE = histoExtension;
  const std::string & hX = "DUSG_reject_vs_B_reject_at_cEff_";
	
  // create histograms from base name and extension as given from user
  // BINNING MUST BE IDENTICAL FOR ALL OF THEM!!
  HistoProviderDQM prov("Btag",label_,ibook);
  DUSG_reject_vs_B_reject_at_cEff    = (prov.book2D ( hX + "0_3"    + hE , hX + "0_3"    + hE , nBinOutputX , startOutputX , endOutputX, nBinOutputY , startOutputY , endOutputY )) ;
  DUSG_reject_vs_B_reject_at_cEff->setEfficiencyFlag();
	
  DUSG_reject_vs_B_reject_at_cEff->getTH2F()->SetXTitle ( "B Rejection" );
  DUSG_reject_vs_B_reject_at_cEff->getTH2F()->SetYTitle ( "Light Rejection" );
  DUSG_reject_vs_B_reject_at_cEff->getTH2F()->GetXaxis()->SetTitleOffset ( 0.75 );
  DUSG_reject_vs_B_reject_at_cEff->getTH2F()->GetYaxis()->SetTitleOffset ( 0.75 );

  // loop over eff. vs. discriminator cut b-histo and look in which bin the closest entry is;
  // use fact that eff decreases monotonously

  // any of the histos to be created can be taken here:
  MonitorElement * EffFlavVsXEff = DUSG_reject_vs_B_reject_at_cEff;

  const int& nBinX = EffFlavVsXEff->getTH2F()->GetNbinsX();
  const int& nBinY = EffFlavVsXEff->getTH2F()->GetNbinsY();

  //int n_c_effs = 7;
  double Const_C_eff[7] = {0.2,0.25,0.3,0.35,0.4,0.45,0.5};

  for ( int iBinX = 1; iBinX <= nBinX; iBinX++ ) {  // loop over the bins on the x-axis of the histograms to be filled

    //const float& effBinWidthX = EffFlavVsXEff->getTH2F()->GetBinWidth  ( iBinX );
    //const float& effMidX      = EffFlavVsXEff->getTH2F()->GetBinCenter ( iBinX ); // middle of efficiency bin
    //const float& effLeftX     = effMidX - 0.5*effBinWidthX;              // left edge of bin
    //const float& effRightX    = effMidX + 0.5*effBinWidthX;              // right edge of bin

	for ( int iBinY = 1; iBinY <= nBinY; iBinY++ ) {  // loop over the bins on the y-axis of the histograms to be filled

	    //const float& effBinWidthY = EffFlavVsXEff->getTH2F()->GetBinWidth  ( iBinY );
	    //const float& effMidY      = EffFlavVsXEff->getTH2F()->GetBinCenter ( iBinY ); // middle of efficiency bin
	    //const float& effLeftY     = effMidY - 0.5*effBinWidthY;              // left edge of bin
	    //const float& effRightY    = effMidY + 0.5*effBinWidthY;              // right edge of bin
 
    // find the corresponding bin in the efficiency versus discriminator cut histo: closest one in efficiency
    int bin = DUSG_reject_vs_B_reject_at_cEff->getTH2F()->GetBin(iBinX,iBinY,0); //linearized bin number
    cout<<"fixed eff:"<<Const_C_eff[0]<<" bin eff:"<<effVersusDiscr_c->GetBinContent(bin)<<endl;
    cout<<"bin err:"<<effVersusDiscr_c->GetBinError(bin)<<endl;
    
    cout<<"abs(fixed eff. - bin eff.):"<<abs(Const_C_eff[0]-effVersusDiscr_c->GetBinContent(bin))<<" 0.05*fixed eff:"<<Const_C_eff[0]*0.05<<endl; 
    if (abs(Const_C_eff[0]-effVersusDiscr_c->GetBinContent(bin)) > Const_C_eff[0]*0.05) continue;
    cout<<"abs(fixed eff. - bin eff.)/bin err:"<<abs(Const_C_eff[0]-effVersusDiscr_c->GetBinContent(bin))/effVersusDiscr_c->GetBinError(bin)<<endl;
    if (abs(Const_C_eff[0]-effVersusDiscr_c->GetBinContent(bin))/effVersusDiscr_c->GetBinError(bin)>0.5) continue;
    cout<<"bin err:"<<effVersusDiscr_c->GetBinError(bin)<<" 0.5*bin eff:"<<0.5*effVersusDiscr_c->GetBinContent(bin)<<endl;  
    if (effVersusDiscr_c->GetBinError(bin)>0.5*effVersusDiscr_c->GetBinContent(bin)) continue;
    /*
    int binClosest = findBinClosestYValue ( effVersusDiscr_b , effBMid , effBLeft , effBRight );
    if(!doCTagPlots_){
    binClosest = findBinClosestYValue ( effVersusDiscr_b , effBMid , effBLeft , effBRight );
    }
    else{
    binClosest = findBinClosestYValue ( effVersusDiscr_c , effBMid , effBLeft , effBRight );
    }
    
    const bool&  binFound   = ( binClosest > 0 ) ;
    //
    if ( binFound ) {
      // fill the histos
      if(mcPlots_>2){
	EffFlavVsXEff_d    -> Fill ( effBMid , effVersusDiscr_d   ->GetBinContent ( binClosest ) );
	EffFlavVsXEff_u    -> Fill ( effBMid , effVersusDiscr_u   ->GetBinContent ( binClosest ) );
	EffFlavVsXEff_s    -> Fill ( effBMid , effVersusDiscr_s   ->GetBinContent ( binClosest ) );
	EffFlavVsXEff_g    -> Fill ( effBMid , effVersusDiscr_g   ->GetBinContent ( binClosest ) );
	EffFlavVsXEff_dus  -> Fill ( effBMid , effVersusDiscr_dus ->GetBinContent ( binClosest ) );
      }
      EffFlavVsXEff_c    -> Fill ( effBMid , effVersusDiscr_c   ->GetBinContent ( binClosest ) );
      EffFlavVsXEff_b    -> Fill ( effBMid , effVersusDiscr_b   ->GetBinContent ( binClosest ) );
      EffFlavVsXEff_ni   -> Fill ( effBMid , effVersusDiscr_ni  ->GetBinContent ( binClosest ) );
      EffFlavVsXEff_dusg -> Fill ( effBMid , effVersusDiscr_dusg->GetBinContent ( binClosest ) );
      EffFlavVsXEff_pu   -> Fill ( effBMid , effVersusDiscr_pu  ->GetBinContent ( binClosest ) );

      if(mcPlots_>2){
	EffFlavVsXEff_d  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_d   ->GetBinError ( binClosest ) );
	EffFlavVsXEff_u  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_u   ->GetBinError ( binClosest ) );
	EffFlavVsXEff_s  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_s   ->GetBinError ( binClosest ) );
	EffFlavVsXEff_g  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_g   ->GetBinError ( binClosest ) );
	EffFlavVsXEff_dus->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_dus ->GetBinError ( binClosest ) );
      }
      EffFlavVsXEff_c  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_c   ->GetBinError ( binClosest ) );
      EffFlavVsXEff_b  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_b   ->GetBinError ( binClosest ) );
      EffFlavVsXEff_ni ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_ni  ->GetBinError ( binClosest ) );
      EffFlavVsXEff_dusg->getTH1F() -> SetBinError ( iBinB , effVersusDiscr_dusg->GetBinError ( binClosest ) );
      EffFlavVsXEff_pu ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_pu  ->GetBinError ( binClosest ) );
    }
    else {
      //cout << "Did not find right bin for b-efficiency : " << effBMid << endl;
    }
    */
   } 
  }


}


#include <typeinfo>
