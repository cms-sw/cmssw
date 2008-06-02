#ifndef SimpleL3AbsoluteCorrector_h
#define SimpleL3AbsoluteCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL3AbsoluteCorrector {
 public:
  SimpleL3AbsoluteCorrector ();
  SimpleL3AbsoluteCorrector (const std::string& fDataFile);
  virtual ~SimpleL3AbsoluteCorrector ();

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEta (double fPt, double fEta) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  double correctionBandPtEta (unsigned fBand, double fPt, double fEta) const;
  SimpleJetCorrectorParameters* mParameters;
};

#endif


