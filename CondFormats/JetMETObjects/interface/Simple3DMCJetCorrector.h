//
// Original Author:  Fedor Ratnikov Oct 31, 2007
// $Id: Simple3DMCJetCorrector.h,v 1.2 2007/11/13 23:52:38 fedor Exp $
//
// Standalone 3D MC Jet Corrector
//
#ifndef Simple3DMCJetCorrector_h
#define Simple3DMCJetCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class Simple3DMCJetCorrector {
 public:
  Simple3DMCJetCorrector ();
  Simple3DMCJetCorrector (const std::string& fDataFile);
  virtual ~Simple3DMCJetCorrector ();

  virtual double correctionXYZTEmfraction (double fPx, double fPy, double fPz, double fE, double fEmFraction) const;
  virtual double correctionPtEtaEmfraction (double fPt, double fEta, double fEmFraction) const;
  virtual double correctionEtEtaPhiPEmfraction (double fEt, double fEta, double fPhi, double fP, double fEmFraction) const;

 private:
  Simple3DMCJetCorrector (const Simple3DMCJetCorrector&);
  Simple3DMCJetCorrector& operator= (const Simple3DMCJetCorrector&);
  SimpleJetCorrectorParameters* mParameters;
};

#endif
