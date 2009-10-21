#ifndef JetCorrectionUncertainty_h
#define JetCorrectionUncertainty_h

#include <string>
#include <vector>
class SimpleJetCorrectorParameters;

class JetCorrectionUncertainty {
 public:
  JetCorrectionUncertainty ();
  JetCorrectionUncertainty (const std::string& fDataFile);
  virtual ~JetCorrectionUncertainty ();

  void setParameters (const std::string& fDataFile);
  virtual double uncertaintyXYZT (double fPx, double fPy, double fPz, double fE, std::string fDirection) const;
  virtual double uncertaintyPtEta (double fPt, double fEta, std::string fDirection) const;
  virtual double uncertaintyEtEtaPhiP (double fEt, double fEta, double fPhi, double fP, std::string fDirection) const;

 private:
  JetCorrectionUncertainty (const JetCorrectionUncertainty&);
  JetCorrectionUncertainty& operator= (const JetCorrectionUncertainty&);
  int findPtBin(std::vector<double> v, double fPt) const;
  double uncertaintyBandPtEta (unsigned fBand, double fPt, double fEta, std::string fDirection) const;
  double quadraticInterpolation (double fVar, const double fBinMiddle[3], const double fBinValue[3]) const; 
  double linearInterpolation (double fVar, const double x1, const double x2, const double y1, const double y2) const;
  SimpleJetCorrectorParameters* mParameters;
};

#endif

