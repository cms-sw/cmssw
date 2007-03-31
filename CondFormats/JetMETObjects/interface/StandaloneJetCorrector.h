//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: StandaloneJetCorrector.h,v 1.1 2007/03/30 23:47:53 fedor Exp $
//
// Generic interface for JetCorrection services
//
#ifndef StandaloneJetCorrector_h
#define StandaloneJetCorrector_h

#include <string>

class StandaloneJetCorrector
{
 public:
  StandaloneJetCorrector (){};
  virtual ~StandaloneJetCorrector (){};

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const = 0;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const = 0;
};

#endif
