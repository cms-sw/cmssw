#include "DQMOffline/RecoB/interface/BTagDifferentialPlot.h"
#include "DQMOffline/RecoB/interface/EffPurFromHistos.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TF1.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 

#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"

#include <algorithm>
#include <iostream>
#include <sstream>
using namespace std ;



BTagDifferentialPlot::BTagDifferentialPlot (const double& bEff, const ConstVarType& constVariable,
	const std::string & tagName) :
	fixedBEfficiency     ( bEff )  ,
	noProcessing         ( false ) , processed(false), constVar(constVariable),
	constVariableName    ( "" )    , diffVariableName     ( "" )    ,
	constVariableValue   ( 999.9 , 999.9 ) , commonName( "MisidForBEff_" + tagName+"_") ,
	theDifferentialHistoB_d    ( 0 ) ,
	theDifferentialHistoB_u    ( 0 ) ,
	theDifferentialHistoB_s    ( 0 ) ,
	theDifferentialHistoB_c    ( 0 ) ,
	theDifferentialHistoB_b    ( 0 ) ,
	theDifferentialHistoB_g    ( 0 ) ,
	theDifferentialHistoB_ni   ( 0 ) ,
	theDifferentialHistoB_dus  ( 0 ) ,
	theDifferentialHistoB_dusg ( 0 )  {}


BTagDifferentialPlot::~BTagDifferentialPlot () {
}






void BTagDifferentialPlot::plot (TCanvas & thePlotCanvas ) {

//   thePlotCanvas = new TCanvas(  commonName ,
// 				commonName ,
// 				btppXCanvas , btppYCanvas ) ;
//
//   if ( !btppTitle ) gStyle->SetOptTitle ( 0 ) ;

  if (!processed) return;
//fixme:
  bool btppNI = false;
  bool btppColour = true;

  thePlotCanvas.SetFillColor ( 0 ) ;
  thePlotCanvas.cd ( 1 ) ;
  gPad->SetLogy  ( 1 ) ;
  gPad->SetGridx ( 1 ) ;
  gPad->SetGridy ( 1 ) ;

//  int col_b   ;
  int col_c   ;
  int col_g   ;
  int col_dus ;
  int col_ni  ;

//  int mStyle_b   ;
  int mStyle_c   ;
  int mStyle_g   ;
  int mStyle_dus ;
  int mStyle_ni  ;

  // marker size (same for all)
  float mSize = 1.5 ;

  if ( btppColour ) {
//    col_b    = 2 ;
    col_c    = 6 ;
    col_g    = 3 ;
    col_dus  = 4 ;
    col_ni   = 5 ;
//    mStyle_b   = 20 ;
    mStyle_c   = 20 ;
    mStyle_g   = 20 ;
    mStyle_dus = 20 ;
    mStyle_ni  = 20 ;
  }
  else {
//    col_b    = 1 ;
    col_c    = 1 ;
    col_g    = 1 ;
    col_dus  = 1 ;
    col_ni   = 1 ;
//    mStyle_b   = 12 ;
    mStyle_c   = 22 ;
    mStyle_g   = 29 ;
    mStyle_dus = 20 ;
    mStyle_ni  = 27 ;
  }

  // for the moment: plot b (to see what the constant b-efficiency is), c, g, uds
  // b in red
  // No, do not plot b (because only visible for the soft leptons)
  // theDifferentialHistoB_b   -> GetXaxis()->SetTitle ( diffVariableName ) ;
  // theDifferentialHistoB_b   -> GetYaxis()->SetTitle ( "non b-jet efficiency" ) ;
  // theDifferentialHistoB_b   -> GetYaxis()->SetTitleOffset ( 1.25 ) ;
  // theDifferentialHistoB_b   -> SetMaximum ( 0.4 )  ;
  // theDifferentialHistoB_b   -> SetMinimum ( 1.e-4 )  ;
  // theDifferentialHistoB_b   -> SetMarkerColor ( col_b ) ;
  // theDifferentialHistoB_b   -> SetLineColor   ( col_b ) ;
  // theDifferentialHistoB_b   -> SetMarkerSize  ( mSize ) ;
  // theDifferentialHistoB_b   -> SetMarkerStyle ( mStyle_b ) ;
  // theDifferentialHistoB_b   -> SetStats ( false ) ;
  // theDifferentialHistoB_b   -> Draw ( "pe" ) ;
  // c in magenta
  theDifferentialHistoB_c ->getTH1F()  -> SetMaximum ( 0.4 )  ;
  theDifferentialHistoB_c ->getTH1F()  -> SetMinimum ( 1.e-4 )  ;
  theDifferentialHistoB_c ->getTH1F()  -> SetMarkerColor ( col_c ) ;
  theDifferentialHistoB_c ->getTH1F()  -> SetLineColor   ( col_c ) ;
  theDifferentialHistoB_c ->getTH1F()  -> SetMarkerSize  ( mSize ) ;
  theDifferentialHistoB_c ->getTH1F()  -> SetMarkerStyle ( mStyle_c ) ;
  theDifferentialHistoB_c ->getTH1F()  -> SetStats     ( false ) ;
  //  theDifferentialHistoB_c   -> Draw("peSame") ;
  theDifferentialHistoB_c   ->getTH1F()-> Draw("pe") ;
  // uds in blue
  theDifferentialHistoB_dus ->getTH1F()-> SetMarkerColor ( col_dus ) ;
  theDifferentialHistoB_dus ->getTH1F()-> SetLineColor   ( col_dus ) ;
  theDifferentialHistoB_dus ->getTH1F()-> SetMarkerSize  ( mSize ) ;
  theDifferentialHistoB_dus ->getTH1F()-> SetMarkerStyle ( mStyle_dus ) ;
  theDifferentialHistoB_dus ->getTH1F()-> SetStats     ( false ) ;
  theDifferentialHistoB_dus ->getTH1F()-> Draw("peSame") ;
  // g in green
  // only uds not to confuse
  theDifferentialHistoB_g   ->getTH1F()-> SetMarkerColor ( col_g ) ;
  theDifferentialHistoB_g   ->getTH1F()-> SetLineColor   ( col_g ) ;
  theDifferentialHistoB_g   ->getTH1F()-> SetMarkerSize  ( mSize ) ;
  theDifferentialHistoB_g   ->getTH1F()-> SetMarkerStyle ( mStyle_g ) ;
  theDifferentialHistoB_g   ->getTH1F()-> SetStats     ( false ) ;
  theDifferentialHistoB_g   ->getTH1F()-> Draw("peSame") ;

  // NI if wanted
  if ( btppNI ) {
    theDifferentialHistoB_ni ->getTH1F()-> SetMarkerColor ( col_ni ) ;
    theDifferentialHistoB_ni ->getTH1F()-> SetLineColor   ( col_ni ) ;
    theDifferentialHistoB_ni ->getTH1F()-> SetMarkerSize  ( mSize ) ;
    theDifferentialHistoB_ni ->getTH1F()-> SetMarkerStyle ( mStyle_ni ) ;
    theDifferentialHistoB_ni ->getTH1F()-> SetStats     ( false ) ;
    theDifferentialHistoB_ni ->getTH1F()-> Draw("peSame") ;
  }
}

void BTagDifferentialPlot::epsPlot(const std::string & name)
{
  plot(name, ".eps");
}

void BTagDifferentialPlot::psPlot(const std::string & name)
{
  plot(name, ".ps");
}

void BTagDifferentialPlot::plot(const std::string & name, const std::string & ext)
{
  if (!processed) return;
   TCanvas tc(commonName.c_str(), commonName.c_str());
   plot(tc);
   tc.Print((name + commonName + ext).c_str());
}


void BTagDifferentialPlot::process () {
  setVariableName () ; // also sets noProcessing if not OK
  if ( noProcessing ) return ;
  bookHisto () ;
  fillHisto () ;
  processed = true;
}


void BTagDifferentialPlot::setVariableName ()
{
  if ( constVar==constETA ) {
    constVariableName  = "eta" ;
    diffVariableName   = "pt"  ;
    constVariableValue = make_pair ( theBinPlotters[0]->etaPtBin().getEtaMin() , theBinPlotters[0]->etaPtBin().getEtaMax() ) ;
  }
  if ( constVar==constPT  ) {
    constVariableName = "pt"  ;
    diffVariableName  = "eta" ;
    constVariableValue = make_pair ( theBinPlotters[0]->etaPtBin().getPtMin() , theBinPlotters[0]->etaPtBin().getPtMax() ) ;
  }

  /*  std::cout
     << "====>>>> BTagDifferentialPlot::setVariableName() : set const/diffVariableName to : "
     << constVariableName << " / " << diffVariableName << endl
     << "====>>>>                                            constant value interval : "
     << constVariableValue.first  << " - " << constVariableValue.second << endl ;
  */
}



void BTagDifferentialPlot::bookHisto () {

  // vector with ranges
  vector<float> variableRanges ;

  for ( vector<JetTagPlotter *>::const_iterator iP = theBinPlotters.begin() ; 
        iP != theBinPlotters.end() ; ++iP ) {
    const EtaPtBin & currentBin = (*iP)->etaPtBin()  ;
    if ( diffVariableName == "eta" ) {
      // only active bins in the variable on x-axis
      if ( currentBin.getEtaActive() ) {
	variableRanges.push_back ( currentBin.getEtaMin() ) ;
	// also max if last one
	if ( iP == --theBinPlotters.end() ) variableRanges.push_back ( currentBin.getEtaMax() ) ;
      }
    }
    if ( diffVariableName == "pt" ) {
      // only active bins in the variable on x-axis
      if ( currentBin.getPtActive() ) {
	variableRanges.push_back ( currentBin.getPtMin() ) ;
	// also max if last one
	if ( iP == --theBinPlotters.end() ) variableRanges.push_back ( currentBin.getPtMax() ) ;
      }
    }
  }

  // to book histo with variable binning -> put into array
  int      nBins    = variableRanges.size() - 1 ;
  float * binArray = &variableRanges[0];
  //float * binArray = new float [nBins+1] ;

  //for ( int i = 0 ; i < nBins + 1 ; i++ ) {
  //  binArray[i] = variableRanges[i] ;
  //}


  // part of the name common to all flavours
  std::stringstream stream("");
  stream << fixedBEfficiency << "_Const_" << constVariableName << "_" << constVariableValue.first << "-" ;
  stream << constVariableValue.second << "_" << "_Vs_" << diffVariableName ;
  commonName += stream.str();
  std::remove(commonName.begin(), commonName.end(), ' ') ;
  std::replace(commonName.begin(), commonName.end(), '.' , 'v' ) ;

  std::string label(commonName);
  HistoProviderDQM prov ("Btag",label);

  theDifferentialHistoB_d    = (prov.book1D ( "D_"    + commonName , "D_"    + commonName , nBins , binArray )) ;
  theDifferentialHistoB_u    = (prov.book1D ( "U_"    + commonName , "U_"    + commonName , nBins , binArray )) ;
  theDifferentialHistoB_s    = (prov.book1D ( "S_"    + commonName , "S_"    + commonName , nBins , binArray )) ;
  theDifferentialHistoB_c    = (prov.book1D ( "C_"    + commonName , "C_"    + commonName , nBins , binArray )) ;
  theDifferentialHistoB_b    = (prov.book1D ( "B_"    + commonName , "B_"    + commonName , nBins , binArray )) ;
  theDifferentialHistoB_g    = (prov.book1D ( "G_"    + commonName , "G_"    + commonName , nBins , binArray )) ;
  theDifferentialHistoB_ni   = (prov.book1D ( "NI_"   + commonName , "NI_"   + commonName , nBins , binArray )) ;
  theDifferentialHistoB_dus  = (prov.book1D ( "DUS_"  + commonName , "DUS_"  + commonName , nBins , binArray )) ;
  theDifferentialHistoB_dusg = (prov.book1D ( "DUSG_" + commonName , "DUSG_" + commonName , nBins , binArray )) ;
}


void BTagDifferentialPlot::fillHisto () {
  // loop over bins and find corresponding misid. in the MisIdVs..... histo
  for ( vector<JetTagPlotter *>::const_iterator iP = theBinPlotters.begin() ;
        iP != theBinPlotters.end() ; ++iP ) {
    const EtaPtBin   & currentBin              = (*iP)->etaPtBin() ;
    EffPurFromHistos * currentEffPurFromHistos = (*iP)->getEffPurFromHistos() ;
    //
    bool   isActive   = true ;
    double valueXAxis = -999.99 ;
    // find right bin based on middle of the interval
    if ( diffVariableName == "eta" ) {
      isActive = currentBin.getEtaActive() ;
      valueXAxis = 0.5 * ( currentBin.getEtaMin() + currentBin.getEtaMax() ) ;
    } else if ( diffVariableName == "pt"  ) {
      isActive = currentBin.getPtActive() ;
      valueXAxis = 0.5 * ( currentBin.getPtMin() + currentBin.getPtMax() ) ;
    } else {
      throw cms::Exception("Configuration")
	<< "====>>>> BTagDifferentialPlot::fillHisto() : illegal diffVariableName = " << diffVariableName << endl;
    }

    // for the moment: ignore inactive bins
    // (maybe later: if a Bin is inactive -> set value to fill well below left edge of histogram to have it in the underflow)

    if ( !isActive ) continue ;

    // to have less lines of code ....
    vector< pair<TH1F*,TH1F*> > effPurDifferentialPairs ;

    // all flavours (b is a good cross check! must be constant and = fixed b-efficiency)
    // get histo; find the bin of the fixed b-efficiency in the histo and get misid; fill


    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_d()    , theDifferentialHistoB_d ->getTH1F()   ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_u()    , theDifferentialHistoB_u ->getTH1F()   ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_s()    , theDifferentialHistoB_s ->getTH1F()   ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_c()    , theDifferentialHistoB_c  ->getTH1F()  ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_b()    , theDifferentialHistoB_b  ->getTH1F()  ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_g()    , theDifferentialHistoB_g  ->getTH1F()  ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_ni()   , theDifferentialHistoB_ni ->getTH1F()  ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_dus()  , theDifferentialHistoB_dus->getTH1F()  ) ) ;
    effPurDifferentialPairs.push_back ( make_pair ( currentEffPurFromHistos->getEffFlavVsBEff_dusg() , theDifferentialHistoB_dusg->getTH1F() ) ) ;

    for ( vector< pair<TH1F*,TH1F*> >::const_iterator itP  = effPurDifferentialPairs.begin() ;
	                                              itP != effPurDifferentialPairs.end()   ; ++itP ) {
      TH1F * effPurHist = itP->first  ;
      TH1F * diffHist   = itP->second ;
      pair<double, double> mistag = getMistag(fixedBEfficiency, effPurHist);
      int iBinSet = diffHist->FindBin(valueXAxis) ;
      diffHist->SetBinContent(iBinSet, mistag.first);
      diffHist->SetBinError(iBinSet, mistag.second);
    }
  }

}

pair<double, double>
BTagDifferentialPlot::getMistag(const double& fixedBEfficiency, TH1F * effPurHist)
{
  int iBinGet = effPurHist->FindBin ( fixedBEfficiency ) ;
  double effForBEff    = effPurHist->GetBinContent ( iBinGet ) ;
  double effForBEffErr = effPurHist->GetBinError   ( iBinGet ) ;

  if (effForBEff==0. && effForBEffErr==0.) {
    // The bin was empty. Could be that it was not filled, as the scan-plot
    //  did not have an entry at the requested value, or that the curve
    // is above or below.
    // Fit a plynomial, and evaluate the mistag at the requested value.
    int fitStatus;
    try {
      fitStatus = effPurHist->Fit("pol4", "q");
    }catch (cms::Exception& iException){
      return pair<double, double>(effForBEff, effForBEffErr);
    }
    if (fitStatus != 0) {
      edm::LogWarning("BTagDifferentialPlot")<<"Fit failed to hisogram " << effPurHist->GetTitle() << " , perhaps because too few entries = " << effPurHist->GetEntries() <<". This bin will be missing in plots at fixed b efficiency.";
      //    } else {
      //      edm::LogInfo("BTagDifferentialPlot")<<"Fit OK to hisogram " << effPurHist->GetTitle() << " entries = " << effPurHist->GetEntries();
      return pair<double, double>(effForBEff, effForBEffErr);
    }
    TF1 *myfunc = effPurHist->GetFunction("pol4");
    effForBEff = myfunc->Eval(fixedBEfficiency);
    effPurHist->RecursiveRemove(myfunc);
    //Error: first non-empty bin on the right and take its error
    for (int i = iBinGet+1; i< effPurHist->GetNbinsX(); ++i) {
      if (effPurHist->GetBinContent(i)!=0) {
        effForBEffErr = effPurHist->GetBinError(i);
	break;
      }
    }
  }

  return pair<double, double>(effForBEff, effForBEffErr);
}

