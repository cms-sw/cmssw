#ifndef SimpleL3PFAbsoluteCorrector_h
#define SimpleL3PFAbsoluteCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL3PFAbsoluteCorrector {
 public:
  SimpleL3PFAbsoluteCorrector ();
  SimpleL3PFAbsoluteCorrector (const std::string& fDataFile);
  virtual ~SimpleL3PFAbsoluteCorrector ();

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEta (double fPt, double fEta) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  SimpleL3PFAbsoluteCorrector (const SimpleL3PFAbsoluteCorrector&);
  SimpleL3PFAbsoluteCorrector& operator= (const SimpleL3PFAbsoluteCorrector&);
  double correctionBandPtEta (unsigned fBand, double fPt, double fEta) const;
  SimpleJetCorrectorParameters* mParameters;
};

#endif


