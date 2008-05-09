//
//
// TOOLS declarations
//
//

#ifndef Tools_H
#define Tools_H

#include "TH1F.h"
#include "TArrayF.h"
#include "TObjArray.h"
#include "TCanvas.h"
#include "TString.h"
#include "TFile.h"
class TStyle;

#include<iostream>
#include<string>
#include<sstream>


namespace RecoBTag {
  double HistoBinWidth ( TH1F * theHisto , int iBin ) ;

  double IntegrateHistogram ( TH1F * theHisto ) ;

  void HistoToNormalizedArrays ( TH1F * theHisto , TArrayF & theNormalizedArray , TArrayF & theLeftOfBinArray , TArrayF & theBinWidthArray ) ;

  double IntegrateArray ( const TArrayF & theArray , const TArrayF & theBinWidth ) ;

  void PrintHistos ( TString psFile , TString epsFile , TString gifFile ) ;

  void PrintCanvasHistos ( TCanvas * canvas , TString psFile , TString epsFile , TString gifFile ) ;

  TObjArray getHistArray ( TFile * histoFile , TString baseName ) ;

  TString flavour ( const int flav ) ;

  bool flavourIsD ( const int & flav )    ;
  bool flavourIsU ( const int & flav )    ;
  bool flavourIsS ( const int & flav )    ;
  bool flavourIsC ( const int & flav )    ;
  bool flavourIsB ( const int & flav )    ;
  bool flavourIsG ( const int & flav )    ;
  bool flavourIsDUS  ( const int & flav ) ;
  bool flavourIsDUSG ( const int & flav ) ;
  bool flavourIsNI  ( const int & flav )  ;

  int  checkCreateDirectory ( TString ) ;

  int findBinClosestYValue ( TH1F * , float yVal , float yLow , float yHigh ) ;

  TStyle* setTDRStyle();

  void tdrGrid(bool gridOn);

  void fixOverlay();

  std::string itos(int i);	// convert int to string

}
#endif
