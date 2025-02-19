#ifndef PixelTrackFitting_PixelTrackErrorParam_H
#define PixelTrackFitting_PixelTrackErrorParam_H
#include <cmath>

class PixelTrackErrorParam {
public: 
 PixelTrackErrorParam(double eta, double pt);
 double errPt()  const;
// { return ptPar0_[theIEta]*thePt + ptPar1_[theIEta]*thePt2; }
 inline double errCot() const { return cotPar0_[theIEta] + cotPar1_[theIEta]/thePt + cotPar2_[theIEta]/thePt2; }
 inline double errTip() const { return sqrt(tipPar0_[theIEta] + tipPar1_[theIEta]/thePt2); }
 inline double errZip() const { return sqrt(zipPar0_[theIEta] + zipPar1_[theIEta]/thePt2); } 
 inline double errPhi() const { return sqrt(phiPar0_[theIEta] + phiPar1_[theIEta]/thePt2); } 

private:
  unsigned int theIEta;
  double thePt, thePt2;

  static const unsigned int nEta = 25;
  static double ptPar0_[nEta], ptPar1_[nEta];
  static double cotPar0_[nEta], cotPar1_[nEta], cotPar2_[nEta];
  static double tipPar0_[nEta], tipPar1_[nEta];
  static double zipPar0_[nEta], zipPar1_[nEta];
  static double phiPar0_[nEta], phiPar1_[nEta];
};

#endif
