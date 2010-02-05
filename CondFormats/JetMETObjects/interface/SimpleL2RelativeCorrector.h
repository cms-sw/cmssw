#ifndef SimpleL2RelativeCorrector_h
#define SimpleL2RelativeCorrector_h

#include <string>

class SimpleJetCorrectorParameters;

class SimpleL2RelativeCorrector {
 public:
  SimpleL2RelativeCorrector ();
  SimpleL2RelativeCorrector (const std::string& fDataFile);
  virtual ~SimpleL2RelativeCorrector ();

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEta (double fPt, double fEta) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  SimpleL2RelativeCorrector (const SimpleL2RelativeCorrector&);
  SimpleL2RelativeCorrector& operator= (const SimpleL2RelativeCorrector&);
  double correctionBandPtEta (unsigned fBand, double fPt, double fEta) const;
  double quadraticInterpolation (double fEta, const double fEtaMiddle[3], const double fEtaValue[3]) const; 
  SimpleJetCorrectorParameters* mParameters;
};

#endif


