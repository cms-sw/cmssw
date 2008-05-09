#ifndef BaseBTagPlotter_H
#define BaseBTagPlotter_H

#include "DQMOffline/RecoB/interface/EtaPtBin.h"
#include "DQMOffline/RecoB/interface/EffPurFromHistos.h"
#include "TString.h"

class BaseBTagPlotter {

 public:

  BaseBTagPlotter ( const TString & tagName, const EtaPtBin & etaPtBin) :
	etaPtBin_(etaPtBin), tagName_(tagName),
	theExtensionString ("_"+tagName+etaPtBin.getDescriptionString()) {};

  virtual ~BaseBTagPlotter () {};
  
  const EtaPtBin& etaPtBin() { return etaPtBin_ ;}
  
  // final computation, plotting, printing .......
  virtual void finalize () = 0;

  virtual void epsPlot(const TString & name) = 0;

  virtual void psPlot(const TString & name) = 0;

//   int nBinEffPur() const {return nBinEffPur_;}
//   double startEffPur() const {return startEffPur_;}
//   double endEffPur() const {return endEffPur_;}
// 
 protected:

  // the extension string to be used in histograms etc.
  const EtaPtBin etaPtBin_;
  const TString tagName_, theExtensionString,  ;
//   const int   nBinEffPur_ ;
//   const double startEffPur_ ; 
//   const double endEffPur_ ; 
  
  
} ;

#endif
