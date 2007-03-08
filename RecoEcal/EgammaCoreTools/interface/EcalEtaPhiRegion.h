#ifndef RecoEcal_EgammaCoreTools_EcalEtaPhiRegion_h
#define RecoEcal_EgammaCoreTools_EcalEtaPhiRegion_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class EcalEtaPhiRegion
{
 public:

  EcalEtaPhiRegion(double etaLow, double etaHigh, double phiLow, double phiHigh);
  ~EcalEtaPhiRegion() {};

  double etaLow() const { return etaLow_; }
  double etaHigh() const { return etaHigh_; }
  double phiLow() const { return phiLow_; }
  double phiHigh() const { return phiHigh_; }

  bool inRegion(const GlobalPoint& position) const;

 private:

  double etaLow_;
  double etaHigh_;
  double phiLow_;
  double phiHigh_;

};

#endif
