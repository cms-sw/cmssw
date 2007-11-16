//
// Original Author:  Fedor Ratnikov Oct 31, 2007
// $Id: SimpleL5FlavorCorrector.h,v 1.1 2007/11/01 21:50:30 fedor Exp $
//
// Jet flavor dependent corrections
//
#ifndef SimpleL5FlavorCorrector_h
#define SimpleL5FlavorCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL5FlavorCorrector {
 public:
  SimpleL5FlavorCorrector ();
  SimpleL5FlavorCorrector (const std::string& fDataFile);
  virtual ~SimpleL5FlavorCorrector ();

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEta (double fPt, double fEta) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  double correctionBandPtEta (unsigned fBand, double fPt, double fEta) const;
  SimpleJetCorrectorParameters* mParameters;
};

#endif
