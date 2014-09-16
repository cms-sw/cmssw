
#ifndef FlavourHistograms_H
#define FlavourHistograms_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 

// #include "BTagPlotPrintC.h"

#include "TH1F.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"

#include "DQMOffline/RecoB/interface/Tools.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"
#include <iostream>
#include <string>

//class DQMStore;

//
// class to describe Histo
//
template <class T>
class FlavourHistograms {

public:

  FlavourHistograms (const std::string& baseNameTitle_ , const std::string& baseNameDescription_ ,
		     const int& nBins_ , const double& lowerBound_ , const double& upperBound_ ,
		     const bool& statistics_ , const bool& plotLog_ , const bool& plotNormalized_ ,
		     const std::string& plotFirst_ , const bool& update, const std::string& folder, 
		     const unsigned int& mc, DQMStore::IBooker & ibook) ;

  virtual ~FlavourHistograms () ;


  // define arrays (if needed)
//   void defineArray ( int * dimension , int max , int indexToPlot ) ;
  
  // fill entry
  // For single variables and arrays (for arrays only a single index can be filled)
  void fill ( const int & flavour,  const T & variable) const;
  void fill ( const int & flavour,  const T & variable, const T & w) const;

  // For single variables and arrays
  void fill ( const int & flavour,  const T * variable) const;


  void settitle(const char* title) ;
  
  void plot (TPad * theCanvas = 0) ;

  void epsPlot(const std::string& name);

  // needed for efficiency computations -> this / b
  // (void : alternative would be not to overwrite the histos but to return a cloned HistoDescription)
  void divide ( const FlavourHistograms<T> & bHD ) const ;

  inline void SetMaximum(const double& max) { theMax = max;}
  inline void SetMinimum(const double& min) { theMin = min;}


  // trivial access functions
  inline std::string  baseNameTitle       () const { return theBaseNameTitle       ; }
  inline std::string  baseNameDescription () const { return theBaseNameDescription ; }
  inline int    nBins               () const { return theNBins               ; }
  inline double lowerBound          () const { return theLowerBound          ; } 
  inline double upperBound          () const { return theUpperBound          ; }
  inline bool   statistics          () const { return theStatistics          ; }
  inline bool   plotLog             () const { return thePlotLog             ; }
  inline bool   plotNormalized      () const { return thePlotNormalized      ; }
  inline std::string  plotFirst           () const { return thePlotFirst           ; }
  inline int* arrayDimension () const { return theArrayDimension; }
  inline int maxDimension () const {return theMaxDimension; }
  inline int indexToPlot () const {return theIndexToPlot; }

  // access to the histos
  inline TH1F * histo_all  () const { return theHisto_all->getTH1F()  ; }    
  inline TH1F * histo_d    () const { return theHisto_d ->getTH1F()   ; }    
  inline TH1F * histo_u    () const { return theHisto_u->getTH1F()    ; }
  inline TH1F * histo_s    () const { return theHisto_s->getTH1F()    ; }
  inline TH1F * histo_c    () const { return theHisto_c->getTH1F()    ; }
  inline TH1F * histo_b    () const { return theHisto_b->getTH1F()    ; }
  inline TH1F * histo_g    () const { return theHisto_g->getTH1F()    ; }
  inline TH1F * histo_ni   () const { return theHisto_ni->getTH1F()   ; }
  inline TH1F * histo_dus  () const { return theHisto_dus->getTH1F()  ; }
  inline TH1F * histo_dusg () const { return theHisto_dusg->getTH1F() ; }
  inline TH1F * histo_pu () const { return theHisto_pu->getTH1F() ; }

  std::vector<TH1F*> getHistoVector() const;
  

protected:

  void fillVariable ( const int & flavour , const T & var , const T & w) const;
  
  //
  // the data members
  //
  
//   T *   theVariable   ;

  // for arrays
  int * theArrayDimension ;
  int   theMaxDimension ;
  int   theIndexToPlot ; // in case that not the complete array has to be plotted

  std::string  theBaseNameTitle ;
  std::string  theBaseNameDescription ;
  int    theNBins ;
  double theLowerBound ;
  double theUpperBound ;
  bool   theStatistics ;
  bool   thePlotLog    ;
  bool   thePlotNormalized ;
  std::string  thePlotFirst  ; // one character; gives flavour to plot first: l (light) , c , b
  double theMin, theMax;

  // the histos
  MonitorElement *theHisto_all  ;    
  MonitorElement *theHisto_d    ;    
  MonitorElement *theHisto_u    ;
  MonitorElement *theHisto_s    ;
  MonitorElement *theHisto_c    ;
  MonitorElement *theHisto_b    ;
  MonitorElement *theHisto_g    ;
  MonitorElement *theHisto_ni   ;
  MonitorElement *theHisto_dus  ;
  MonitorElement *theHisto_dusg ;
  MonitorElement *theHisto_pu ;

  //  DQMStore * dqmStore_; 


  // the canvas to plot
  TCanvas* theCanvas ;
  private:
  FlavourHistograms(){}

  unsigned int mcPlots_;

} ;

template <class T>
FlavourHistograms<T>::FlavourHistograms (const std::string& baseNameTitle_ , const std::string& baseNameDescription_ ,
					 const int& nBins_ , const double& lowerBound_ , const double& upperBound_ ,
					 const bool& statistics_ , const bool& plotLog_ , const bool& plotNormalized_ ,
					 const std::string& plotFirst_, const bool& update, const std::string& folder, 
					 const unsigned int& mc, DQMStore::IBooker & ibook) :
  // BaseFlavourHistograms () ,
  // theVariable ( variable_ ) ,
  theMaxDimension(-1), theIndexToPlot(-1), theBaseNameTitle ( baseNameTitle_ ) , theBaseNameDescription ( baseNameDescription_ ) ,
  theNBins ( nBins_ ) , theLowerBound ( lowerBound_ ) , theUpperBound ( upperBound_ ) ,
  theStatistics ( statistics_ ) , thePlotLog ( plotLog_ ) , thePlotNormalized ( plotNormalized_ ) ,
  thePlotFirst ( plotFirst_ ), theMin(-1.), theMax(-1.), mcPlots_(mc)
{
  // defaults for array dimensions
  theArrayDimension = 0  ;
    
  // check plot order string 
  if ( thePlotFirst == "l" || thePlotFirst == "c" || thePlotFirst == "b" ) {
    // OK
  }
  else {
    // not correct: print warning and set default (l)
    std::cout << "FlavourHistograms::FlavourHistograms : thePlotFirst was not correct : " << thePlotFirst << std::endl ;
    std::cout << "FlavourHistograms::FlavourHistograms : Set it to default value (l)! " << std::endl ;
    thePlotFirst = "l" ;
  }

  if (!update) {
    // book histos
    HistoProviderDQM prov("Btag",folder,ibook);
    if(mcPlots_%2==0) theHisto_all   = (prov.book1D( theBaseNameTitle + "ALL"  , theBaseNameDescription + " all jets"  , theNBins , theLowerBound , theUpperBound )) ; 
    else theHisto_all = 0;
    if (mcPlots_) {
      if(mcPlots_>2) {
	theHisto_d     = (prov.book1D ( theBaseNameTitle + "D"    , theBaseNameDescription + " d-jets"    , theNBins , theLowerBound , theUpperBound )) ; 
	theHisto_u     = (prov.book1D ( theBaseNameTitle + "U"    , theBaseNameDescription + " u-jets"    , theNBins , theLowerBound , theUpperBound )) ; 
	theHisto_s     = (prov.book1D ( theBaseNameTitle + "S"    , theBaseNameDescription + " s-jets"    , theNBins , theLowerBound , theUpperBound )) ; 
	theHisto_g     = (prov.book1D ( theBaseNameTitle + "G"    , theBaseNameDescription + " g-jets"    , theNBins , theLowerBound , theUpperBound )) ; 
	theHisto_dus   = (prov.book1D ( theBaseNameTitle + "DUS"  , theBaseNameDescription + " dus-jets"  , theNBins , theLowerBound , theUpperBound )) ; 
      }
      else{
	theHisto_d = 0;
	theHisto_u = 0;
	theHisto_s = 0;
	theHisto_g = 0;
	theHisto_dus = 0;
      }
      theHisto_c     = (prov.book1D ( theBaseNameTitle + "C"    , theBaseNameDescription + " c-jets"    , theNBins , theLowerBound , theUpperBound )) ; 
      theHisto_b     = (prov.book1D ( theBaseNameTitle + "B"    , theBaseNameDescription + " b-jets"    , theNBins , theLowerBound , theUpperBound )) ; 
      theHisto_ni    = (prov.book1D ( theBaseNameTitle + "NI"   , theBaseNameDescription + " ni-jets"   , theNBins , theLowerBound , theUpperBound )) ; 
      theHisto_dusg  = (prov.book1D ( theBaseNameTitle + "DUSG" , theBaseNameDescription + " dusg-jets" , theNBins , theLowerBound , theUpperBound )) ;
      theHisto_pu    = (prov.book1D ( theBaseNameTitle + "PU"   , theBaseNameDescription + " pu-jets"   , theNBins , theLowerBound , theUpperBound )) ; 
    }else{
      theHisto_d = 0;
      theHisto_u = 0;
      theHisto_s = 0;
      theHisto_c = 0;
      theHisto_b = 0;
      theHisto_g = 0;
      theHisto_ni = 0;
      theHisto_dus = 0;
      theHisto_dusg = 0;
      theHisto_pu = 0;
    }

      // statistics if requested
    if ( theStatistics ) {
      if(theHisto_all) theHisto_all ->getTH1F()->Sumw2() ; 
      if (mcPlots_ ) {  
	if (mcPlots_>2 ) {
	  theHisto_d   ->getTH1F()->Sumw2() ; 
	  theHisto_u   ->getTH1F()->Sumw2() ; 
	  theHisto_s   ->getTH1F()->Sumw2() ;
	  theHisto_g   ->getTH1F()->Sumw2() ; 
	  theHisto_dus ->getTH1F()->Sumw2() ; 
	} 
	theHisto_c   ->getTH1F()->Sumw2() ; 
	theHisto_b   ->getTH1F()->Sumw2() ; 
	theHisto_ni  ->getTH1F()->Sumw2() ; 
	theHisto_dusg->getTH1F()->Sumw2() ;
	theHisto_pu  ->getTH1F()->Sumw2() ;
      }
    }
  } else {
    //is it useful? anyway access function is deprecated... 
    HistoProviderDQM prov("Btag",folder,ibook);
    if(theHisto_all) theHisto_all   = prov.access(theBaseNameTitle + "ALL" ) ; 
    if (mcPlots_) {  
      if (mcPlots_>2 ) {
	theHisto_d     = prov.access(theBaseNameTitle + "D"   ) ; 
	theHisto_u     = prov.access(theBaseNameTitle + "U"   ) ; 
	theHisto_s     = prov.access(theBaseNameTitle + "S"   ) ;
	theHisto_g     = prov.access(theBaseNameTitle + "G"   ) ; 
	theHisto_dus   = prov.access(theBaseNameTitle + "DUS" ) ; 
      } 
      theHisto_c     = prov.access(theBaseNameTitle + "C"   ) ; 
      theHisto_b     = prov.access(theBaseNameTitle + "B"   ) ; 
      theHisto_ni    = prov.access(theBaseNameTitle + "NI"  ) ; 
      theHisto_dusg  = prov.access(theBaseNameTitle + "DUSG") ;
      theHisto_pu  = prov.access(theBaseNameTitle + "PU") ;
    }
  }

  // defaults for other data members
  theCanvas = 0 ;
}


template <class T>
FlavourHistograms<T>::~FlavourHistograms () {
  // delete the canvas*/
  delete theCanvas ;
}


// define arrays (if needed)
// template <class T>
// void FlavourHistograms<T>::defineArray ( int * dimension , int max , int indexToPlot ) {
//   // indexToPlot < 0 if all to be plotted
//   theArrayDimension = dimension ;
//   theMaxDimension   = max ;
//   theIndexToPlot    = indexToPlot ;
// }
  
// fill entry
template <class T> void
FlavourHistograms<T>::fill ( const int & flavour,  const T & variable) const 
{
  // For single variables and arrays (for arrays only a single index can be filled)
  fillVariable ( flavour , variable, 1.) ;
}

template <class T> void
FlavourHistograms<T>::fill ( const int & flavour,  const T & variable, const T & w) const 
{
  // For single variables and arrays (for arrays only a single index can be filled)
  fillVariable ( flavour , variable, w) ;
}

template <class T> void
FlavourHistograms<T>::fill ( const int & flavour,  const T * variable) const
{
  if ( theArrayDimension == 0 ) {       
    // single variable
    fillVariable ( flavour , *variable, 1.) ;
  } else {
    // array      
    int iMax = (*theArrayDimension > theMaxDimension) ? theMaxDimension : *theArrayDimension ;
    //
    for ( int i = 0 ; i != iMax ; ++i ) {
      // check if only one index to be plotted (<0: switched off -> plot all)
      if ( ( theIndexToPlot < 0 ) || ( i == theIndexToPlot ) ) { 
	fillVariable ( flavour , *(variable+i) , 1.) ;
      }
    }

    // if single index to be filled but not enough entries: fill 0.0 (convention!)
    if ( theIndexToPlot >= iMax ) { 
      // cout << "==>> The index to be filled is too big -> fill 0.0 : " << theBaseNameTitle << " : " << theIndexToPlot << " >= " << iMax << endl ;
      const T& theZero = static_cast<T> ( 0.0 ) ;
      fillVariable ( flavour , theZero , 1.) ;
    }
  }
} 


template <class T>
void FlavourHistograms<T>::settitle(const char* title) {
 if(theHisto_all) theHisto_all ->setAxisTitle(title) ;
    if (mcPlots_) {  
      if (mcPlots_>2 ) {
	theHisto_d   ->setAxisTitle(title) ;
	theHisto_u   ->setAxisTitle(title) ;
	theHisto_s   ->setAxisTitle(title) ;
	theHisto_g   ->setAxisTitle(title) ;
	theHisto_dus ->setAxisTitle(title) ;
      }
      theHisto_c   ->setAxisTitle(title) ;
      theHisto_b   ->setAxisTitle(title) ;
      theHisto_ni  ->setAxisTitle(title) ;
      theHisto_dusg->setAxisTitle(title) ;
      theHisto_pu->setAxisTitle(title) ;
    }
}


template <class T>
void FlavourHistograms<T>::plot (TPad * theCanvas /* = 0 */) {

//fixme:
  bool btppNI = false;
  bool btppColour = true;

  if (theCanvas)
    theCanvas->cd();
  
  RecoBTag::setTDRStyle()->cd();
  gPad->UseCurrentStyle();
//   if ( !btppTitle ) gStyle->SetOptTitle ( 0 ) ;
//   
//   // here: plot histograms in a canvas
//   theCanvas = new TCanvas ( "C" + theBaseNameTitle , "C" + theBaseNameDescription ,
// 			    btppXCanvas , btppYCanvas ) ;
//   theCanvas->SetFillColor ( 0 ) ;
//   theCanvas->cd  ( 1 ) ;
  gPad->SetLogy  ( 0 ) ;
  if ( thePlotLog ) gPad->SetLogy ( 1 ) ; 
  gPad->SetGridx ( 0 ) ;
  gPad->SetGridy ( 0 ) ;
  gPad->SetTitle ( 0 ) ;

  MonitorElement * histo[4];
  int col[4], lineStyle[4], markerStyle[4];
  int lineWidth = 1 ;

  const double markerSize = gPad->GetWh() * gPad->GetHNDC() / 500.;

  // default (l)
  histo[0] = theHisto_dusg ;
  //CW histo_1 = theHisto_dus ;
  histo[1] = theHisto_b ;
  histo[2] = theHisto_c ;
  histo[3]= 0 ;

  double max = theMax;
  if (theMax<=0.) {
    max = theHisto_dusg->getTH1F()->GetMaximum();
    if (theHisto_b->getTH1F()->GetMaximum() > max) max = theHisto_b->getTH1F()->GetMaximum();
    if (theHisto_c->getTH1F()->GetMaximum() > max) max = theHisto_c->getTH1F()->GetMaximum();
  }

  if (btppNI) {
    histo[3] = theHisto_ni ;
    if (theHisto_ni->getTH1F()->GetMaximum() > max) max = theHisto_ni->getTH1F()->GetMaximum();
  }

  if ( btppColour ) { // print colours 
    col[0] = 4 ;
    col[1] = 2 ;
    col[2] = 6 ;
    col[3] = 3 ;
    lineStyle[0] = 1 ;
    lineStyle[1] = 1 ;
    lineStyle[2] = 1 ;
    lineStyle[3] = 1 ;
    markerStyle[0] = 20 ;
    markerStyle[1] = 21 ;
    markerStyle[2] = 22 ;
    markerStyle[3] = 23 ;
   lineWidth = 1 ;
  }
  else { // different marker/line styles
    col[1] = 1 ;
    col[2] = 1 ;
    col[3] = 1 ;
    col[0] = 1 ;
    lineStyle[0] = 2 ;
    lineStyle[1] = 1 ;
    lineStyle[2] = 3 ;
    lineStyle[3] = 4 ;
    markerStyle[0] = 20 ;
    markerStyle[1] = 21 ;
    markerStyle[2] = 22 ;
    markerStyle[3] = 23 ;
  }

  // if changing order (NI stays always last)
  
  // c to plot first   
  if ( thePlotFirst == "c" ) {
    histo[0] = theHisto_c ;
    if ( btppColour  ) col[0] = 6 ;
    if ( !btppColour ) lineStyle[0] = 3 ;
    histo[2] = theHisto_dusg ;
    if ( btppColour  ) col[2] = 4 ;
    if ( !btppColour ) lineStyle[2] = 2 ;
  }

  // b to plot first   
  if ( thePlotFirst == "b" ) {
    histo[0] = theHisto_b ;
    if ( btppColour  ) col[0] = 2 ;
    if ( !btppColour ) lineStyle[0] = 1 ;
    histo[1] = theHisto_dusg ;
    if ( btppColour  ) col[1] = 4 ;
    if ( !btppColour ) lineStyle[1] = 2 ;
  }


  histo[0] ->getTH1F()->GetXaxis()->SetTitle ( theBaseNameDescription.c_str() ) ;
  histo[0] ->getTH1F()->GetYaxis()->SetTitle ( "Arbitrary Units" ) ;
  histo[0] ->getTH1F()->GetYaxis()->SetTitleOffset(1.25) ;

  for (int i=0; i != 4; ++i) {
    if (histo[i]== 0 ) continue;
    histo[i] ->getTH1F()->SetStats ( false ) ;
    histo[i] ->getTH1F()->SetLineStyle ( lineStyle[i] ) ;
    histo[i] ->getTH1F()->SetLineWidth ( lineWidth ) ;
    histo[i] ->getTH1F()->SetLineColor ( col[i] ) ;
    histo[i] ->getTH1F()->SetMarkerStyle ( markerStyle[i] ) ;
    histo[i] ->getTH1F()->SetMarkerColor ( col[i] ) ;
    histo[i] ->getTH1F()->SetMarkerSize ( markerSize ) ;
  }

  if ( thePlotNormalized ) {
//   cout <<histo[0]->GetEntries() <<" "<< histo[1]->GetEntries() <<" "<< histo[2]->GetEntries()<<" "<<histo[3]->GetEntries()<<endl;
    if (histo[0]->getTH1F()->GetEntries() != 0) {histo[0]->getTH1F() ->DrawNormalized() ;} else {    histo[0]->getTH1F() ->SetMaximum(1.0);
histo[0] ->getTH1F()->Draw() ;}
    if (histo[1]->getTH1F()->GetEntries() != 0) histo[1] ->getTH1F()->DrawNormalized("Same") ;
    if (histo[2]->getTH1F()->GetEntries() != 0) histo[2]->getTH1F() ->DrawNormalized("Same") ;
    if ((histo[3] != 0) && (histo[3]->getTH1F()->GetEntries() != 0))  histo[3] ->getTH1F()->DrawNormalized("Same") ;
  }
  else {
    histo[0]->getTH1F()->SetMaximum(max*1.05);
    if (theMin!=-1.) histo[0]->getTH1F()->SetMinimum(theMin);
    histo[0]->getTH1F()->Draw() ;
    histo[1]->getTH1F()->Draw("Same") ;
    histo[2]->getTH1F()->Draw("Same") ;
    if ( histo[3] != 0 ) histo[3]->getTH1F()->Draw("Same") ;
  }

}

template <class T>
void FlavourHistograms<T>::epsPlot(const std::string& name)
{
   TCanvas tc(theBaseNameTitle.c_str() , theBaseNameDescription.c_str());
   
   plot(&tc);
   tc.Print((name + theBaseNameTitle + ".eps").c_str());
}


// needed for efficiency computations -> this / b
// (void : alternative would be not to overwrite the histos but to return a cloned HistoDescription)
template <class T>
void FlavourHistograms<T>::divide ( const FlavourHistograms<T> & bHD ) const {
  // divide histos using binomial errors
  //
  // ATTENTION: It's the responsability of the user to make sure that the HistoDescriptions
  //            involved in this operation have been constructed with the statistics option switched on!!
  //
  if(theHisto_all) theHisto_all  ->getTH1F()-> Divide ( theHisto_all->getTH1F()  , bHD.histo_all () , 1.0 , 1.0 , "b" ) ;    
    if (mcPlots_) {
      if (mcPlots_>2 ) {
	theHisto_d    ->getTH1F()-> Divide ( theHisto_d ->getTH1F()   , bHD.histo_d   () , 1.0 , 1.0 , "b" ) ;    
	theHisto_u    ->getTH1F()-> Divide ( theHisto_u ->getTH1F()   , bHD.histo_u   () , 1.0 , 1.0 , "b" ) ;
	theHisto_s    ->getTH1F()-> Divide ( theHisto_s ->getTH1F()   , bHD.histo_s   () , 1.0 , 1.0 , "b" ) ;
	theHisto_g    ->getTH1F()-> Divide ( theHisto_g  ->getTH1F()  , bHD.histo_g   () , 1.0 , 1.0 , "b" ) ;
	theHisto_dus  ->getTH1F()-> Divide ( theHisto_dus->getTH1F()  , bHD.histo_dus () , 1.0 , 1.0 , "b" ) ;
      }
      theHisto_c    ->getTH1F()-> Divide ( theHisto_c ->getTH1F()   , bHD.histo_c   () , 1.0 , 1.0 , "b" ) ;
      theHisto_b    ->getTH1F()-> Divide ( theHisto_b ->getTH1F()   , bHD.histo_b   () , 1.0 , 1.0 , "b" ) ;
      theHisto_ni   ->getTH1F()-> Divide ( theHisto_ni->getTH1F()   , bHD.histo_ni  () , 1.0 , 1.0 , "b" ) ;
      theHisto_dusg ->getTH1F()-> Divide ( theHisto_dusg->getTH1F() , bHD.histo_dusg() , 1.0 , 1.0 , "b" ) ;
      theHisto_pu   ->getTH1F()-> Divide ( theHisto_pu->getTH1F() , bHD.histo_pu() , 1.0 , 1.0 , "b" ) ;
    }
}
  

template <class T>
void FlavourHistograms<T>::fillVariable ( const int & flavour , const T & var , const T & w) const {

  // all, except for the Jet Multiplicity which is not filled for each jets but for each events  
  if( (theBaseNameDescription != "Jet Multiplicity" || flavour == -1) && theHisto_all) theHisto_all->Fill ( var ,w) ;

  // flavour specific
  if (!mcPlots_ || (theBaseNameDescription == "Jet Multiplicity" && flavour == -1)) return;

  switch(flavour) {
    case 1:
      if (mcPlots_>2 ) {
	theHisto_d->Fill( var ,w);
	if(theBaseNameDescription != "Jet Multiplicity") theHisto_dus->Fill( var ,w);
      }
      if(theBaseNameDescription != "Jet Multiplicity") theHisto_dusg->Fill( var ,w);
      return;
    case 2:
      if (mcPlots_>2 ) {
	theHisto_u->Fill( var ,w);
	if(theBaseNameDescription != "Jet Multiplicity") theHisto_dus->Fill( var ,w);
      }
      if(theBaseNameDescription != "Jet Multiplicity") theHisto_dusg->Fill( var ,w);
      return;
    case 3:
      if (mcPlots_>2 ) {
	theHisto_s->Fill( var ,w);
	if(theBaseNameDescription != "Jet Multiplicity") theHisto_dus->Fill( var ,w);
      }
      if(theBaseNameDescription != "Jet Multiplicity") theHisto_dusg->Fill( var ,w);
      return;
    case 4:
      theHisto_c->Fill( var ,w);
      return;
    case 5:
      theHisto_b->Fill( var ,w);
      return;
    case 21:
      if (mcPlots_>2 ) theHisto_g->Fill( var ,w);
      if(theBaseNameDescription != "Jet Multiplicity") theHisto_dusg->Fill( var ,w);
      return;
    case 123:
      if (mcPlots_>2 && theBaseNameDescription == "Jet Multiplicity") theHisto_dus->Fill( var ,w);
      return;
    case 12321:
      if (theBaseNameDescription == "Jet Multiplicity") theHisto_dusg->Fill( var ,w);
      return;
    case 20:
      theHisto_pu->Fill( var ,w);
      return;
    default:
      theHisto_ni->Fill( var ,w);
      return;
  }
}

template <class T>
std::vector<TH1F*> FlavourHistograms<T>::getHistoVector() const
{
  std::vector<TH1F*> histoVector;
  if(theHisto_all) histoVector.push_back ( theHisto_all->getTH1F() );
    if (mcPlots_) {  
      if (mcPlots_>2 ) {
	histoVector.push_back ( theHisto_d->getTH1F()   );
	histoVector.push_back ( theHisto_u->getTH1F()   );
	histoVector.push_back ( theHisto_s->getTH1F()   );
	histoVector.push_back ( theHisto_g ->getTH1F()  );
	histoVector.push_back ( theHisto_dus->getTH1F() );
      }
      histoVector.push_back ( theHisto_c->getTH1F()   );
      histoVector.push_back ( theHisto_b->getTH1F()   );
      histoVector.push_back ( theHisto_ni->getTH1F()  );
      histoVector.push_back ( theHisto_dusg->getTH1F());
      histoVector.push_back ( theHisto_pu->getTH1F());
    }
  return histoVector;
}
#endif
