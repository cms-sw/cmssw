//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: JetCorrector.h,v 1.3 2007/01/18 01:35:13 fedor Exp $
//
// Generic interface for JetCorrection services
//
#ifndef SimpleJetCorrector_h
#define SimpleJetCorrector_h

#include <string>

class SimpleJetCorrector
{
 public:
  SimpleJetCorrector (){};
  virtual ~SimpleJetCorrector (){};

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE, ) const = 0;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE, ) const = 0;
};

#endif
