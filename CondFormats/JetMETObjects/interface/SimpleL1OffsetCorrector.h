#ifndef SimpleL1OffsetCorrector_h
#define SimpleL1OffsetCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL1OffsetCorrector {
 public:
  SimpleL1OffsetCorrector ();
  SimpleL1OffsetCorrector (const std::string& fDataFile);
  virtual ~SimpleL1OffsetCorrector ();

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionEnEta (double fE, double fEta) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  SimpleL1OffsetCorrector (const SimpleL1OffsetCorrector&);
  SimpleL1OffsetCorrector& operator= (const SimpleL1OffsetCorrector&);
  double correctionBandEnEta (unsigned fBand, double fE, double fEta) const;
  SimpleJetCorrectorParameters* mParameters;
};

#endif


