//
// Original Author:  Fedor Ratnikov Oct 31, 2007
// $Id: Simple3DMCJetCorrector.h,v 1.3 2007/03/31 18:24:16 fedor Exp $
//
// Standalone 3D MC Jet Corrector
//
#ifndef Simple3DMCJetCorrector_h
#define Simple3DMCJetCorrector_h

#include <string>

class Simple3DMCJetCorrectorParameters;

class Simple3DMCJetCorrector {
 public:
  Simple3DMCJetCorrector ();
  Simple3DMCJetCorrector (const std::string& fDataFile);
  virtual ~Simple3DMCJetCorrector ();

  virtual double correctionXYZTEmfraction (double fPx, double fPy, double fPz, double fE, double fEmFraction) const;
  virtual double correctionPtEtaPhiEEmfraction (double fPt, double fEta, double fPhi, double fE, double fEmFraction) const;
  virtual double correctionEtEtaPhiPEmfraction (double fEt, double fEta, double fPhi, double fP, double fEmFraction) const;

 private:
  Simple3DMCJetCorrectorParameters* mParameters;
};

#endif
