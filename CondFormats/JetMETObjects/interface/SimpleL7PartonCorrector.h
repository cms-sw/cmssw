//
// Original Author:  Attilio Santocchia Feb 28, 2008
//
// Jet Parton dependent corrections
//
#ifndef SimpleL7PartonCorrector_h
#define SimpleL7PartonCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL7PartonCorrector {
 public:
  SimpleL7PartonCorrector ();
  SimpleL7PartonCorrector (const std::string& fDataFile, const std::string& fSection);
  virtual ~SimpleL7PartonCorrector ();

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEta (double fPt, double fEta) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  SimpleL7PartonCorrector (const SimpleL7PartonCorrector&);
  SimpleL7PartonCorrector& operator= (const SimpleL7PartonCorrector&);
  SimpleJetCorrectorParameters* mParameters;
};

#endif
