
#ifndef FlavourHistograms2D_H
#define FlavourHistograms2D_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 

// #include "BTagPlotPrintC.h"

#include "TString.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"

#include "DQMOffline/RecoB/interface/Tools.h"
#include "DQMOffline/RecoB/interface/HistoProviderDQM.h"

#include <iostream>
#include <vector>
//class DQMStore;

using namespace std ;
using namespace RecoBTag;
//
// class to describe Histo
//
template <class T, class G>
class FlavourHistograms2D {

public:

  FlavourHistograms2D (TString baseNameTitle_ , TString baseNameDescription_ ,
		      int nBinsX_ , double lowerBoundX_ , double upperBoundX_ ,
                      int nBinsY_ , double lowerBoundY_ , double upperBoundY_ ,
		      bool statistics_ ,
		      bool update, std::string folder, bool mc) ;

  virtual ~FlavourHistograms2D () ;


  // define arrays (if needed)
//   void defineArray ( int * dimension , int max , int indexToPlot ) ;
  
  // fill entry
  // For single variables and arrays (for arrays only a single index can be filled)
  void fill ( const int & flavour,  const T & variableX, const G & variableY) const;

  // For single variables and arrays
  void fill ( const int & flavour,  const T * variableX, const G * variableY) const;


  void settitle(const char* title) ;
  
  // needed for efficiency computations -> this / b
  // (void : alternative would be not to overwrite the histos but to return a cloned HistoDescription)
  void divide ( FlavourHistograms2D<T, G> & bHD ) const ;

  void SetMaximum(const double max) { theMax = max;}
  void SetMinimum(const double min) { theMin = min;}


  // trivial access functions
  TString  baseNameTitle       () const { return theBaseNameTitle       ; }
  TString  baseNameDescription () const { return theBaseNameDescription ; }
  int    nBinsX              () const { return theNBinsX              ; }
  int    nBinsY              () const { return theNBinsY              ; }
  double lowerBoundX         () const { return theLowerBoundX         ; } 
  double upperBoundX         () const { return theUpperBoundX         ; }
  double lowerBoundY         () const { return theLowerBoundY         ; } 
  double upperBoundY         () const { return theUpperBoundY         ; }
  bool   statistics          () const { return theStatistics          ; }

  // access to the histos
  TH2F * histo_all  () const { return theHisto_all->getTH2F()  ; }    
  TH2F * histo_d    () const { return theHisto_d ->getTH2F()   ; }    
  TH2F * histo_u    () const { return theHisto_u->getTH2F()    ; }
  TH2F * histo_s    () const { return theHisto_s->getTH2F()    ; }
  TH2F * histo_c    () const { return theHisto_c->getTH2F()    ; }
  TH2F * histo_b    () const { return theHisto_b->getTH2F()    ; }
  TH2F * histo_g    () const { return theHisto_g->getTH2F()    ; }
  TH2F * histo_ni   () const { return theHisto_ni->getTH2F()   ; }
  TH2F * histo_dus  () const { return theHisto_dus->getTH2F()  ; }
  TH2F * histo_dusg () const { return theHisto_dusg->getTH2F() ; }

  std::vector<TH2F*> getHistoVector() const;

  

protected:

  void fillVariable ( const int & flavour , const T & varX , const G & varY ) const;
  
  //
  // the data members
  //
  
//   T *   theVariable   ;

  // for arrays
  int * theArrayDimension ;
  int   theMaxDimension ;
  int   theIndexToPlot ; // in case that not the complete array has to be plotted

  TString  theBaseNameTitle ;
  TString  theBaseNameDescription ;
  int    theNBinsX ;
  int    theNBinsY ;
  double theLowerBoundX ;
  double theUpperBoundX ;
  double theLowerBoundY ;
  double theUpperBoundY ;
  bool   theStatistics ;
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

  //  DQMStore * dqmStore_; 


  // the canvas to plot
  private:
  FlavourHistograms2D(){}

  bool mcPlots_;

} ;



template <class T, class G>
FlavourHistograms2D<T, G>::FlavourHistograms2D (TString baseNameTitle_ , TString baseNameDescription_ ,
					  int nBinsX_ , double lowerBoundX_ , double upperBoundX_ ,
                                          int nBinsY_ , double lowerBoundY_ , double upperBoundY_ ,
					  bool statistics_ ,
					  bool update, std::string folder, bool mc) :
  // BaseFlavourHistograms2D () ,
  // theVariable ( variable_ ) ,
  theBaseNameTitle ( baseNameTitle_ ) , theBaseNameDescription ( baseNameDescription_ ) ,
  theNBinsX ( nBinsX_ ) , theNBinsY (nBinsY_), 
  theLowerBoundX ( lowerBoundX_ ) , theUpperBoundX ( upperBoundX_ ) ,
  theLowerBoundY ( lowerBoundY_ ) , theUpperBoundY ( upperBoundY_ ) ,
  theStatistics ( statistics_ ) ,
  theMin(-1.), theMax(-1.), mcPlots_(mc)
{
  // defaults for array dimensions
  theArrayDimension = 0  ;
  theMaxDimension   = -1 ;
  theIndexToPlot    = -1 ;
    
  if (!update) {
    // book histos
    HistoProviderDQM prov("Btag",folder);
    theHisto_all   = (prov.book2D( theBaseNameTitle + "ALL"  , theBaseNameDescription + " all jets"  , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY )) ; 
    if (mcPlots_ == true) {  
      theHisto_d     = (prov.book2D ( theBaseNameTitle + "D"    , theBaseNameDescription + " d-jets"    , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY )) ; 
      theHisto_u     = (prov.book2D ( theBaseNameTitle + "U"    , theBaseNameDescription + " u-jets"    , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_s     = (prov.book2D ( theBaseNameTitle + "S"    , theBaseNameDescription + " s-jets"    , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_c     = (prov.book2D ( theBaseNameTitle + "C"    , theBaseNameDescription + " c-jets"    , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_b     = (prov.book2D ( theBaseNameTitle + "B"    , theBaseNameDescription + " b-jets"    , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_g     = (prov.book2D ( theBaseNameTitle + "G"    , theBaseNameDescription + " g-jets"    , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_ni    = (prov.book2D ( theBaseNameTitle + "NI"   , theBaseNameDescription + " ni-jets"   , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_dus   = (prov.book2D ( theBaseNameTitle + "DUS"  , theBaseNameDescription + " dus-jets"  , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ; 
      theHisto_dusg  = (prov.book2D ( theBaseNameTitle + "DUSG" , theBaseNameDescription + " dusg-jets" , theNBinsX , theLowerBoundX , theUpperBoundX , theNBinsY, theLowerBoundY, theUpperBoundY)) ;
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
    }
      // statistics if requested
    if ( theStatistics ) {
      theHisto_all ->getTH2F()->Sumw2() ; 
      if (mcPlots_ == true) {  
	
	theHisto_d   ->getTH2F()->Sumw2() ; 
	theHisto_u   ->getTH2F()->Sumw2() ; 
	theHisto_s   ->getTH2F()->Sumw2() ; 
	theHisto_c   ->getTH2F()->Sumw2() ; 
	theHisto_b   ->getTH2F()->Sumw2() ; 
	theHisto_g   ->getTH2F()->Sumw2() ; 
	theHisto_ni  ->getTH2F()->Sumw2() ; 
	theHisto_dus ->getTH2F()->Sumw2() ; 
	theHisto_dusg->getTH2F()->Sumw2() ;
      }
    }
  } else {
    HistoProviderDQM prov("Btag",folder);
    theHisto_all   = prov.access(theBaseNameTitle + "ALL" ) ; 
    if (mcPlots_ == true) {  
      
      theHisto_d     = prov.access(theBaseNameTitle + "D"   ) ; 
      theHisto_u     = prov.access(theBaseNameTitle + "U"   ) ; 
      theHisto_s     = prov.access(theBaseNameTitle + "S"   ) ; 
      theHisto_c     = prov.access(theBaseNameTitle + "C"   ) ; 
      theHisto_b     =prov.access(theBaseNameTitle + "B"   ) ; 
      theHisto_g     =prov.access(theBaseNameTitle + "G"   ) ; 
      theHisto_ni    =prov.access(theBaseNameTitle + "NI"  ) ; 
      theHisto_dus   =prov.access(theBaseNameTitle + "DUS" ) ; 
      theHisto_dusg  =prov.access(theBaseNameTitle + "DUSG") ;
    }
  }
}


template <class T, class G>
FlavourHistograms2D<T, G>::~FlavourHistograms2D () {}


// define arrays (if needed)
// template <class T, class G>
// void FlavourHistograms2D<T, G>::defineArray ( int * dimension , int max , int indexToPlot ) {
//   // indexToPlot < 0 if all to be plotted
//   theArrayDimension = dimension ;
//   theMaxDimension   = max ;
//   theIndexToPlot    = indexToPlot ;
// }
  
// fill entry
template <class T, class G> void
FlavourHistograms2D<T, G>::fill ( const int & flavour,  const T & variableX, const G & variableY) const 
{
  // For single variables and arrays (for arrays only a single index can be filled)
  fillVariable ( flavour , variableX , variableY ) ;
}

template <class T, class G> void
FlavourHistograms2D<T, G>::fill ( const int & flavour,  const T * variableX, const G * variableY) const
{
  if ( theArrayDimension == 0 ) {       
    // single variable
    fillVariable ( flavour , *variableX, *variableY ) ;
  } else {
    // array      
    int iMax = *theArrayDimension ;
    if ( *theArrayDimension > theMaxDimension ) iMax = theMaxDimension ;
    //
    for ( int i = 0 ; i < iMax ; i++ ) {
      // check if only one index to be plotted (<0: switched off -> plot all)
      if ( ( theIndexToPlot < 0 ) || ( i == theIndexToPlot ) ) { 
	fillVariable ( flavour , *(variableX+i) , *(variableY+i) ) ;
      }
    }

    // if single index to be filled but not enough entries: fill 0.0 (convention!)
    if ( theIndexToPlot >= iMax ) { 
      // cout << "==>> The index to be filled is too big -> fill 0.0 : " << theBaseNameTitle << " : " << theIndexToPlot << " >= " << iMax << endl ;
      T theZeroT = static_cast<T> ( 0.0) ;
      G theZeroG = static_cast<T> ( 0.0 );
      fillVariable ( flavour , theZeroT , theZeroG ) ;
    }
  }
} 


template <class T, class G>
void FlavourHistograms2D<T, G>::settitle(const char* title) {
 if(theHisto_all) theHisto_all ->setAxisTitle(title) ;
    if (mcPlots_ == true) {  
      
      if(theHisto_d)  theHisto_d   ->setAxisTitle(title) ;
      if(theHisto_u)  theHisto_u   ->setAxisTitle(title) ;
      if(theHisto_s)  theHisto_s   ->setAxisTitle(title) ;
      if(theHisto_c)  theHisto_c   ->setAxisTitle(title) ;
      if(theHisto_b)  theHisto_b   ->setAxisTitle(title) ;
      if(theHisto_g)  theHisto_g   ->setAxisTitle(title) ;
      if(theHisto_ni) theHisto_ni  ->setAxisTitle(title) ;
      if(theHisto_dus) theHisto_dus ->setAxisTitle(title) ;
      if(theHisto_dusg)theHisto_dusg->setAxisTitle(title) ;
    }
}

// needed for efficiency computations -> this / b
// (void : alternative would be not to overwrite the histos but to return a cloned HistoDescription)
template <class T, class G>
void FlavourHistograms2D<T, G>::divide ( FlavourHistograms2D<T, G> & bHD ) const {
  // divide histos using binomial errors
  //
  // ATTENTION: It's the responsability of the user to make sure that the HistoDescriptions
  //            involved in this operation have been constructed with the statistics option switched on!!
  //
  theHisto_all  ->getTH2F()-> Divide ( theHisto_all->getTH2F()  , bHD.histo_all () , 1.0 , 1.0 , "b" ) ;    
    if (mcPlots_ == true) {  
      theHisto_d    ->getTH2F()-> Divide ( theHisto_d ->getTH2F()   , bHD.histo_d   () , 1.0 , 1.0 , "b" ) ;    
      theHisto_u    ->getTH2F()-> Divide ( theHisto_u ->getTH2F()   , bHD.histo_u   () , 1.0 , 1.0 , "b" ) ;
      theHisto_s    ->getTH2F()-> Divide ( theHisto_s ->getTH2F()   , bHD.histo_s   () , 1.0 , 1.0 , "b" ) ;
      theHisto_c    ->getTH2F()-> Divide ( theHisto_c ->getTH2F()   , bHD.histo_c   () , 1.0 , 1.0 , "b" ) ;
      theHisto_b    ->getTH2F()-> Divide ( theHisto_b ->getTH2F()   , bHD.histo_b   () , 1.0 , 1.0 , "b" ) ;
      theHisto_g    ->getTH2F()-> Divide ( theHisto_g  ->getTH2F()  , bHD.histo_g   () , 1.0 , 1.0 , "b" ) ;
      theHisto_ni   ->getTH2F()-> Divide ( theHisto_ni->getTH2F()   , bHD.histo_ni  () , 1.0 , 1.0 , "b" ) ;
      theHisto_dus  ->getTH2F()-> Divide ( theHisto_dus->getTH2F()  , bHD.histo_dus () , 1.0 , 1.0 , "b" ) ;
      theHisto_dusg ->getTH2F()-> Divide ( theHisto_dusg->getTH2F() , bHD.histo_dusg() , 1.0 , 1.0 , "b" ) ;
    }
}
  

template <class T, class G>
void FlavourHistograms2D<T, G>::fillVariable ( const int & flavour , const T & varX , const G & varY ) const {
  // all
  theHisto_all                ->Fill ( varX, varY ) ;
  // flavour specific
  if (mcPlots_ != true) return;

  if ( flavourIsD   (flavour) ) theHisto_d    ->Fill ( varX , varY ) ; 
  if ( flavourIsU   (flavour) ) theHisto_u    ->Fill ( varX , varY ) ; 
  if ( flavourIsS   (flavour) ) theHisto_s    ->Fill ( varX , varY ) ; 
  if ( flavourIsC   (flavour) ) theHisto_c    ->Fill ( varX , varY ) ; 
  if ( flavourIsB   (flavour) ) theHisto_b    ->Fill ( varX , varY ) ; 
  if ( flavourIsG   (flavour) ) theHisto_g    ->Fill ( varX , varY ) ; 
  if ( flavourIsNI  (flavour) ) theHisto_ni   ->Fill ( varX , varY ) ; 
  if ( flavourIsDUS (flavour) ) theHisto_dus  ->Fill ( varX , varY ) ; 
  if ( flavourIsDUSG(flavour) ) theHisto_dusg ->Fill ( varX , varY ) ;
}

template <class T, class G>
vector<TH2F*> FlavourHistograms2D<T, G>::getHistoVector() const
{
  vector<TH2F*> histoVector;
  histoVector.push_back ( theHisto_all->getTH2F() );
    if (mcPlots_ == true) {  
      histoVector.push_back ( theHisto_d->getTH2F()   );
      histoVector.push_back ( theHisto_u->getTH2F()   );
      histoVector.push_back ( theHisto_s->getTH2F()   );
      histoVector.push_back ( theHisto_c->getTH2F()   );
      histoVector.push_back ( theHisto_b->getTH2F()   );
      histoVector.push_back ( theHisto_g ->getTH2F()  );
      histoVector.push_back ( theHisto_ni->getTH2F()  );
      histoVector.push_back ( theHisto_dus->getTH2F() );
      histoVector.push_back ( theHisto_dusg->getTH2F());
    }
  return histoVector;
}
#endif
