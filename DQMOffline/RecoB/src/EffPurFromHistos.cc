#include "DQMOffline/RecoB/interface/EffPurFromHistos.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/CodedException.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <iostream>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 


using namespace std;
using namespace RecoBTag;



EffPurFromHistos::EffPurFromHistos ( const TString & ext, TH1F * h_d, TH1F * h_u,
	TH1F * h_s, TH1F * h_c, TH1F * h_b, TH1F * h_g,	TH1F * h_ni,
	TH1F * h_dus, TH1F * h_dusg,std::string label, int nBin, double startO, double endO) :
	//BTagPlotPrintC(),
	histoExtension(ext), effVersusDiscr_d(h_d), effVersusDiscr_u(h_u),
	effVersusDiscr_s(h_s), effVersusDiscr_c(h_c), effVersusDiscr_b(h_b),
	effVersusDiscr_g(h_g), effVersusDiscr_ni(h_ni), effVersusDiscr_dus(h_dus),
	effVersusDiscr_dusg(h_dusg), nBinOutput(nBin), startOutput(startO),
	endOutput(endO), label_(label)
{
  fromDiscriminatorDistr = false;
  // consistency check
  check();
}

EffPurFromHistos::EffPurFromHistos 
	(const FlavourHistograms<double> * dDiscriminatorFC,std::string label, int nBin,
	double startO, double endO) :
	  nBinOutput(nBin), startOutput(startO), endOutput(endO), label_(label){
  histoExtension = "_"+dDiscriminatorFC->baseNameTitle();


  std::cout << " WRITiNG "<<label<<std::endl;

  fromDiscriminatorDistr = true;
  discrNoCutEffic = new FlavourHistograms<double> (
	"totalEntries" + histoExtension, "Total Entries: " + dDiscriminatorFC->baseNameDescription(),
	dDiscriminatorFC->nBins(), dDiscriminatorFC->lowerBound(),
	dDiscriminatorFC->upperBound(), true, true, false, "b", false, label );

  // conditional discriminator cut for efficiency histos

  discrCutEfficScan = new FlavourHistograms<double> (
	"effVsDiscrCut" + histoExtension, "Eff. vs Disc. Cut: " + dDiscriminatorFC->baseNameDescription(),
	dDiscriminatorFC->nBins(), dDiscriminatorFC->lowerBound(),
	dDiscriminatorFC->upperBound(), true, true, false, "b", false, label );
  discrCutEfficScan->SetMinimum(1E-4);

  effVersusDiscr_d =    discrCutEfficScan->histo_d   ();
  effVersusDiscr_u =    discrCutEfficScan->histo_u   ();
  effVersusDiscr_s =    discrCutEfficScan->histo_s   ();
  effVersusDiscr_c =    discrCutEfficScan->histo_c   ();
  effVersusDiscr_b =    discrCutEfficScan->histo_b   ();
  effVersusDiscr_g =    discrCutEfficScan->histo_g   ();
  effVersusDiscr_ni =   discrCutEfficScan->histo_ni  ();
  effVersusDiscr_dus =  discrCutEfficScan->histo_dus ();
  effVersusDiscr_dusg = discrCutEfficScan->histo_dusg();

  effVersusDiscr_d->SetXTitle ( "Discriminant" );
  effVersusDiscr_d->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_u->SetXTitle ( "Discriminant" );
  effVersusDiscr_u->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_s->SetXTitle ( "Discriminant" );
  effVersusDiscr_s->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_c->SetXTitle ( "Discriminant" );
  effVersusDiscr_c->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_b->SetXTitle ( "Discriminant" );
  effVersusDiscr_b->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_g->SetXTitle ( "Discriminant" );
  effVersusDiscr_g->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_ni->SetXTitle ( "Discriminant" );
  effVersusDiscr_ni->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_dus->SetXTitle ( "Discriminant" );
  effVersusDiscr_dus->GetXaxis()->SetTitleOffset ( 0.75 );
  effVersusDiscr_dusg->SetXTitle ( "Discriminant" );
  effVersusDiscr_dusg->GetXaxis()->SetTitleOffset ( 0.75 );

  // discr. for computation
  vector<TH1F*> discrCfHistos = dDiscriminatorFC->getHistoVector();

  // discr no cut
  vector<TH1F*> discrNoCutHistos = discrNoCutEffic->getHistoVector();

  // discr no cut
  vector<TH1F*> discrCutHistos = discrCutEfficScan->getHistoVector();

  int dimHistos = discrCfHistos.size(); // they all have the same size

  // DISCR-CUT LOOP:
  // fill the histos for eff-pur computations by scanning the discriminatorFC histogram

  // better to loop over bins -> discrCut no longer needed
  int nBins = dDiscriminatorFC->nBins();

  // loop over flavours
  for ( int iFlav = 0; iFlav < dimHistos; iFlav++ ) {
    if (discrCfHistos[iFlav] == 0) continue;
    discrNoCutHistos[iFlav]->SetXTitle ( "Discriminant" );
    discrNoCutHistos[iFlav]->GetXaxis()->SetTitleOffset ( 0.75 );

    // In Root histos, bin counting starts at 1 to nBins.
    // bin 0 is the underflow, and nBins+1 is the overflow.
    double nJetsFlav = discrCfHistos[iFlav]->GetEntries ();
    double sum = discrCfHistos[iFlav]->GetBinContent( nBins+1 ); //+1 to get the overflow.
    
    for ( int iDiscr = nBins; iDiscr > 0 ; --iDiscr ) {
      // fill all jets into NoCut histo
      discrNoCutHistos[iFlav]->SetBinContent ( iDiscr, nJetsFlav );
      discrNoCutHistos[iFlav]->SetBinError   ( iDiscr, sqrt(nJetsFlav) );
      sum += discrCfHistos[iFlav]->GetBinContent( iDiscr );
      discrCutHistos[iFlav]->SetBinContent ( iDiscr, sum );
      discrCutHistos[iFlav]->SetBinError   ( iDiscr, sqrt(sum) );
    }
  }


  // divide to get efficiency vs. discriminator cut from absolute numbers
  discrCutEfficScan->divide ( *discrNoCutEffic );  // does: histos including discriminator cut / flat histo
}


EffPurFromHistos::~EffPurFromHistos () {
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




void EffPurFromHistos::epsPlot(const TString & name)
{
  if ( fromDiscriminatorDistr) {
    discrNoCutEffic->epsPlot(name);
    discrCutEfficScan->epsPlot(name);
  }
  plot(name, ".eps");
}

void EffPurFromHistos::psPlot(const TString & name)
{
  plot(name, ".ps");
}

void EffPurFromHistos::plot(const TString & name, const TString & ext)
{
   TCanvas tc ("FlavEffVsBEff" +histoExtension ,
	"Flavour misidentification vs. b-tagging efficiency " + histoExtension);
   plot(&tc);
   tc.Print(TString(name + "FlavEffVsBEff" + histoExtension + ext));
}

void EffPurFromHistos::plot (TPad * plotCanvas /* = 0 */) {

//fixme:
  bool btppNI = false;
  bool btppColour = true;

//   if ( !btppTitle ) gStyle->SetOptTitle ( 0 );
  setTDRStyle()->cd();

  if (plotCanvas)
    plotCanvas->cd();
  
  gPad->UseCurrentStyle();
  gPad->SetFillColor ( 0 );
  gPad->SetLogy  ( 1 );
  gPad->SetGridx ( 1 );
  gPad->SetGridy ( 1 );

  int col_c  ;
  int col_g  ;
  int col_dus;
  int col_ni ;

  int mStyle_c  ;
  int mStyle_g  ;
  int mStyle_dus;
  int mStyle_ni ;

  // marker size (same for all)
  float mSize = gPad->GetWh() * gPad->GetHNDC() / 500.; //1.2;

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


  // for the moment: plot c,dus,g
  EffFlavVsBEff_dus ->getTH1F()->GetXaxis()->SetTitle ( "b-jet efficiency" );
  EffFlavVsBEff_dus ->getTH1F()->GetYaxis()->SetTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_dus ->getTH1F()->GetYaxis()->SetTitleOffset ( 0.25 );
  EffFlavVsBEff_dus ->getTH1F()->SetMaximum     ( 1.1 );
  EffFlavVsBEff_dus ->getTH1F()->SetMinimum     ( 1.e-5 );
  EffFlavVsBEff_dus ->getTH1F()->SetMarkerColor ( col_dus );
  EffFlavVsBEff_dus ->getTH1F()->SetLineColor   ( col_dus );
  EffFlavVsBEff_dus ->getTH1F()->SetMarkerSize  ( mSize );
  EffFlavVsBEff_dus ->getTH1F()->SetMarkerStyle ( mStyle_dus );
  EffFlavVsBEff_dus ->getTH1F()->SetStats     ( false );
  EffFlavVsBEff_dus ->getTH1F()->Draw("pe");

  EffFlavVsBEff_g   ->getTH1F()->SetMarkerColor ( col_g );
  EffFlavVsBEff_g   ->getTH1F()->SetLineColor   ( col_g );
  EffFlavVsBEff_g   ->getTH1F()->SetMarkerSize  ( mSize );
  EffFlavVsBEff_g   ->getTH1F()->SetMarkerStyle ( mStyle_g );
  EffFlavVsBEff_g   ->getTH1F()->SetStats     ( false );
  EffFlavVsBEff_g   ->getTH1F()->Draw("peSame");

  EffFlavVsBEff_c   ->getTH1F()->SetMarkerColor ( col_c );
  EffFlavVsBEff_c   ->getTH1F()->SetLineColor   ( col_c );
  EffFlavVsBEff_c   ->getTH1F()->SetMarkerSize  ( mSize );
  EffFlavVsBEff_c   ->getTH1F()->SetMarkerStyle ( mStyle_c );
  EffFlavVsBEff_c   ->getTH1F()->SetStats     ( false );
  EffFlavVsBEff_c   ->getTH1F()->Draw("peSame");

  EffFlavVsBEff_d ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_u ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_s ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_c ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_b ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_g ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_ni ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_dus ->getTH1F()-> SetMinimum(0.01);
  EffFlavVsBEff_dusg ->getTH1F()-> SetMinimum(0.01);

  // plot separately u,d and s
//  EffFlavVsBEff_d ->GetXaxis()->SetTitle ( "b-jet efficiency" );
//  EffFlavVsBEff_d ->GetYaxis()->SetTitle ( "non b-jet efficiency" );
//  EffFlavVsBEff_d ->GetYaxis()->SetTitleOffset ( 1.25 );
//  EffFlavVsBEff_d ->SetMaximum     ( 1.1 );
//  EffFlavVsBEff_d ->SetMinimum     ( 1.e-5 );
//  EffFlavVsBEff_d ->SetMarkerColor ( col_dus );
//  EffFlavVsBEff_d ->SetLineColor   ( col_dus );
//  EffFlavVsBEff_d ->SetMarkerSize  ( mSize );
//  EffFlavVsBEff_d ->SetMarkerStyle ( mStyle_dus );
//  EffFlavVsBEff_d ->SetStats     ( false );
//  EffFlavVsBEff_d ->Draw("pe");
//
//  EffFlavVsBEff_u   ->SetMarkerColor ( col_g );
//  EffFlavVsBEff_u   ->SetLineColor   ( col_g );
//  EffFlavVsBEff_u   ->SetMarkerSize  ( mSize );
//  EffFlavVsBEff_u   ->SetMarkerStyle ( mStyle_g );
//  EffFlavVsBEff_u   ->SetStats     ( false );
//  EffFlavVsBEff_u   ->Draw("peSame");
//
//  EffFlavVsBEff_s   ->SetMarkerColor ( col_c );
//  EffFlavVsBEff_s   ->SetLineColor   ( col_c );
//  EffFlavVsBEff_s   ->SetMarkerSize  ( mSize );
//  EffFlavVsBEff_s   ->SetMarkerStyle ( mStyle_c );
//  EffFlavVsBEff_s   ->SetStats     ( false );
//  EffFlavVsBEff_s   ->Draw("peSame");

  // only if asked: NI
  if ( btppNI ) {
    EffFlavVsBEff_ni   ->getTH1F()->SetMarkerColor ( col_ni );
    EffFlavVsBEff_ni   ->getTH1F()->SetLineColor   ( col_ni );
    EffFlavVsBEff_ni   ->getTH1F()->SetMarkerSize  ( mSize );
    EffFlavVsBEff_ni   ->getTH1F()->SetMarkerStyle ( mStyle_ni );
    EffFlavVsBEff_ni   ->getTH1F()->SetStats     ( false );
    EffFlavVsBEff_ni   ->getTH1F()->Draw("peSame");
  }

}


void EffPurFromHistos::check () {
  // number of bins
  int nBins_d    = effVersusDiscr_d    -> GetNbinsX();
  int nBins_u    = effVersusDiscr_u    -> GetNbinsX();
  int nBins_s    = effVersusDiscr_s    -> GetNbinsX();
  int nBins_c    = effVersusDiscr_c    -> GetNbinsX();
  int nBins_b    = effVersusDiscr_b    -> GetNbinsX();
  int nBins_g    = effVersusDiscr_g    -> GetNbinsX();
  int nBins_ni   = effVersusDiscr_ni   -> GetNbinsX();
  int nBins_dus  = effVersusDiscr_dus  -> GetNbinsX();
  int nBins_dusg = effVersusDiscr_dusg -> GetNbinsX();

  bool lNBins =
    ( nBins_d == nBins_u    &&
      nBins_d == nBins_s    &&
      nBins_d == nBins_c    &&
      nBins_d == nBins_b    &&
      nBins_d == nBins_g    &&
      nBins_d == nBins_ni   &&
      nBins_d == nBins_dus  &&
      nBins_d == nBins_dusg     );

  if ( !lNBins ) {
    throw cms::Exception("Configuration")
      << "Input histograms do not all have the same number of bins!\n";
  }


  // start
  float sBin_d    = effVersusDiscr_d    -> GetBinCenter(1);
  float sBin_u    = effVersusDiscr_u    -> GetBinCenter(1);
  float sBin_s    = effVersusDiscr_s    -> GetBinCenter(1);
  float sBin_c    = effVersusDiscr_c    -> GetBinCenter(1);
  float sBin_b    = effVersusDiscr_b    -> GetBinCenter(1);
  float sBin_g    = effVersusDiscr_g    -> GetBinCenter(1);
  float sBin_ni   = effVersusDiscr_ni   -> GetBinCenter(1);
  float sBin_dus  = effVersusDiscr_dus  -> GetBinCenter(1);
  float sBin_dusg = effVersusDiscr_dusg -> GetBinCenter(1);

  bool lSBin =
    ( sBin_d == sBin_u    &&
      sBin_d == sBin_s    &&
      sBin_d == sBin_c    &&
      sBin_d == sBin_b    &&
      sBin_d == sBin_g    &&
      sBin_d == sBin_ni   &&
      sBin_d == sBin_dus  &&
      sBin_d == sBin_dusg     );

  if ( !lSBin ) {
    throw cms::Exception("Configuration")
      << "EffPurFromHistos::check() : Input histograms do not all have the same start bin!\n";
  }


  // end
  float eBin_d    = effVersusDiscr_d    -> GetBinCenter( nBins_d - 1 );
  float eBin_u    = effVersusDiscr_u    -> GetBinCenter( nBins_d - 1 );
  float eBin_s    = effVersusDiscr_s    -> GetBinCenter( nBins_d - 1 );
  float eBin_c    = effVersusDiscr_c    -> GetBinCenter( nBins_d - 1 );
  float eBin_b    = effVersusDiscr_b    -> GetBinCenter( nBins_d - 1 );
  float eBin_g    = effVersusDiscr_g    -> GetBinCenter( nBins_d - 1 );
  float eBin_ni   = effVersusDiscr_ni   -> GetBinCenter( nBins_d - 1 );
  float eBin_dus  = effVersusDiscr_dus  -> GetBinCenter( nBins_d - 1 );
  float eBin_dusg = effVersusDiscr_dusg -> GetBinCenter( nBins_d - 1 );

  bool lEBin =
    ( eBin_d == eBin_u    &&
      eBin_d == eBin_s    &&
      eBin_d == eBin_c    &&
      eBin_d == eBin_b    &&
      eBin_d == eBin_g    &&
      eBin_d == eBin_ni   &&
      eBin_d == eBin_dus  &&
      eBin_d == eBin_dusg     );

  if ( !lEBin ) {
    throw cms::Exception("Configuration")
      << "EffPurFromHistos::check() : Input histograms do not all have the same end bin!\n";
  }
}


void EffPurFromHistos::compute ()
{
  // to have shorter names ......
  TString & hE = histoExtension;
  TString hB = "FlavEffVsBEff_";


  // create histograms from base name and extension as given from user
  // BINNING MUST BE IDENTICAL FOR ALL OF THEM!!
  HistoProviderDQM prov("Btag",label_);
  EffFlavVsBEff_d    = (prov.book1D ( hB + "D"    + hE , hB + "D"    + hE , nBinOutput , startOutput , endOutput ));
  EffFlavVsBEff_u    = (prov.book1D ( hB + "U"    + hE , hB + "U"    + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_s    = (prov.book1D ( hB + "S"    + hE , hB + "S"    + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_c    = (prov.book1D ( hB + "C"    + hE , hB + "C"    + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_b    = (prov.book1D ( hB + "B"    + hE , hB + "B"    + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_g    = (prov.book1D ( hB + "G"    + hE , hB + "G"    + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_ni   = (prov.book1D ( hB + "NI"   + hE , hB + "NI"   + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_dus  = (prov.book1D ( hB + "DUS"  + hE , hB + "DUS"  + hE , nBinOutput , startOutput , endOutput )) ;
  EffFlavVsBEff_dusg = (prov.book1D ( hB + "DUSG" + hE , hB + "DUSG" + hE , nBinOutput , startOutput , endOutput )) ;

  EffFlavVsBEff_d->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_d->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_d->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_d->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_u->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_u->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_u->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_u->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_s->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_s->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_s->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_s->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_c->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_c->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_s->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_s->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_b->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_b->getTH1F()->SetYTitle ( "b-jet efficiency" );
  EffFlavVsBEff_b->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_b->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_g->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_g->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_g->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_g->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_ni->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_ni->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_ni->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_ni->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_dus->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_dus->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_dus->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_dus->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_dusg->getTH1F()->SetXTitle ( "b-jet efficiency" );
  EffFlavVsBEff_dusg->getTH1F()->SetYTitle ( "non b-jet efficiency" );
  EffFlavVsBEff_dusg->getTH1F()->GetXaxis()->SetTitleOffset ( 0.75 );
  EffFlavVsBEff_dusg->getTH1F()->GetYaxis()->SetTitleOffset ( 0.75 );


  // loop over eff. vs. discriminator cut b-histo and look in which bin the closest entry is;
  // use fact that eff decreases monotonously

  // any of the histos to be created can be taken here:
  MonitorElement * EffFlavVsBEff = EffFlavVsBEff_b;

  int nBinB = EffFlavVsBEff->getTH1F()->GetNbinsX();

  for ( int iBinB = 1; iBinB <= nBinB; iBinB++ ) {  // loop over the bins on the x-axis of the histograms to be filled

    float effBBinWidth = EffFlavVsBEff->getTH1F()->GetBinWidth  ( iBinB );
    float effBMid      = EffFlavVsBEff->getTH1F()->GetBinCenter ( iBinB ); // middle of b-efficiency bin
    float effBLeft     = effBMid - 0.5*effBBinWidth;              // left edge of bin
    float effBRight    = effBMid + 0.5*effBBinWidth;              // right edge of bin
    // find the corresponding bin in the efficiency versus discriminator cut histo: closest one in efficiency
    int   binClosest = findBinClosestYValue ( effVersusDiscr_b , effBMid , effBLeft , effBRight );
    bool  binFound   = ( binClosest > 0 ) ;
    //
    if ( binFound ) {
      // fill the histos
      EffFlavVsBEff_d    -> Fill ( effBMid , effVersusDiscr_d   ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_u    -> Fill ( effBMid , effVersusDiscr_u   ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_s    -> Fill ( effBMid , effVersusDiscr_s   ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_c    -> Fill ( effBMid , effVersusDiscr_c   ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_b    -> Fill ( effBMid , effVersusDiscr_b   ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_g    -> Fill ( effBMid , effVersusDiscr_g   ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_ni   -> Fill ( effBMid , effVersusDiscr_ni  ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_dus  -> Fill ( effBMid , effVersusDiscr_dus ->GetBinContent ( binClosest ) );
      EffFlavVsBEff_dusg -> Fill ( effBMid , effVersusDiscr_dusg->GetBinContent ( binClosest ) );

      EffFlavVsBEff_d  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_d   ->GetBinError ( binClosest ) );
      EffFlavVsBEff_u  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_u   ->GetBinError ( binClosest ) );
      EffFlavVsBEff_s  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_s   ->GetBinError ( binClosest ) );
      EffFlavVsBEff_c  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_c   ->GetBinError ( binClosest ) );
      EffFlavVsBEff_b  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_b   ->GetBinError ( binClosest ) );
      EffFlavVsBEff_g  ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_g   ->GetBinError ( binClosest ) );
      EffFlavVsBEff_ni ->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_ni  ->GetBinError ( binClosest ) );
      EffFlavVsBEff_dus->getTH1F()  -> SetBinError ( iBinB , effVersusDiscr_dus ->GetBinError ( binClosest ) );
      EffFlavVsBEff_dusg->getTH1F() -> SetBinError ( iBinB , effVersusDiscr_dusg->GetBinError ( binClosest ) );
    }
    else {
      //CW      cout << "Did not find right bin for b-efficiency : " << effBMid << endl;
    }
    
  }
  
}


#include <typeinfo>

