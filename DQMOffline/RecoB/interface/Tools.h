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
#include "TFile.h"
class TStyle;

#include<iostream>
#include<string>
#include<sstream>


namespace RecoBTag {
  double HistoBinWidth ( const TH1F * theHisto , const int& iBin ) ;

  double IntegrateHistogram ( const TH1F * theHisto ) ;

  void HistoToNormalizedArrays ( const TH1F * theHisto , TArrayF & theNormalizedArray , TArrayF & theLeftOfBinArray , TArrayF & theBinWidthArray ) ;

  double IntegrateArray ( const TArrayF & theArray , const TArrayF & theBinWidth ) ;

  void PrintHistos ( const std::string& psFile , const std::string& epsFile , const std::string& gifFile ) ;

  void PrintCanvasHistos ( TCanvas * canvas , const std::string& psFile , const std::string& epsFile , const std::string& gifFile ) ;

  TObjArray getHistArray ( TFile * histoFile , const std::string& baseName ) ;

  std::string flavour ( const int& flav ) ;

  bool flavourIsD ( const int & flav )    ;
  bool flavourIsU ( const int & flav )    ;
  bool flavourIsS ( const int & flav )    ;
  bool flavourIsC ( const int & flav )    ;
  bool flavourIsB ( const int & flav )    ;
  bool flavourIsG ( const int & flav )    ;
  bool flavourIsDUS  ( const int & flav ) ;
  bool flavourIsDUSG ( const int & flav ) ;
  bool flavourIsNI  ( const int & flav )  ;

  int  checkCreateDirectory ( const std::string& ) ;

  int findBinClosestYValue ( const TH1F * , const float& yVal , const float& yLow , const float& yHigh ) ;

  TStyle* setTDRStyle();

  void tdrGrid(const bool& gridOn);

  void fixOverlay();

  std::string itos(const int& i);	// convert int to string

}
#endif
