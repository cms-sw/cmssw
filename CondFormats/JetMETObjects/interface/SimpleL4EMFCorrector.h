//
// Original Author:  Fedor Ratnikov Oct 31, 2007
// $Id: SimpleL4EMFCorrector.h,v 1.1 2007/11/13 23:52:38 fedor Exp $
//
// Standalone 3D MC Jet Corrector
//
#ifndef SimpleL4EMFCorrector_h
#define SimpleL4EMFCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL4EMFCorrector {
 public:
  SimpleL4EMFCorrector ();
  SimpleL4EMFCorrector (const std::string& fDataFile);
  virtual ~SimpleL4EMFCorrector ();

  virtual double correctionXYZTEmfraction (double fPx, double fPy, double fPz, double fE, double fEmFraction) const;
  virtual double correctionPtEtaEmfraction (double fPt, double fEta, double fEmFraction) const;
  virtual double correctionEtEtaPhiPEmfraction (double fEt, double fEta, double fPhi, double fP, double fEmFraction) const;

 private:
  SimpleL4EMFCorrector (const SimpleL4EMFCorrector&);
  SimpleL4EMFCorrector& operator= (const SimpleL4EMFCorrector&);
  double correctionBandPtEtaEmfraction (unsigned fBand, double fPt, double dEta, double fEmFraction) const;
  SimpleJetCorrectorParameters* mParameters;
};

#endif
