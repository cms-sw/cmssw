#ifndef RecoMuon_TrackerSeedGenerator_L1MuonPixelTrackFitter_H
#define RecoMuon_TrackerSeedGenerator_L1MuonPixelTrackFitter_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>

namespace reco {
  class Track;
}

class TrackingRegion;
class TrackingRecHit;
class L1MuGMTCand;
class PixelRecoLineRZ;
class SeedingHitSet;
class MagneticField;

class L1MuonPixelTrackFitter {
public:
  class Circle {
  public:
    typedef Vector3DBase<long double, GlobalTag> Vector;
    typedef Point3DBase<long double, GlobalTag> Point;
    Circle() : theValid(false) {}
    Circle(const GlobalPoint& h1, const GlobalPoint& h2, double curvature) : theCurvature(curvature) {
      Point p1(h1);
      Point p2(h2);
      Vector dp = (p2 - p1) / 2.;
      int charge = theCurvature > 0 ? 1 : -1;
      Vector ec = charge * dp.cross(Vector(0, 0, 1)).unit();
      long double dist_tmp = 1. / theCurvature / theCurvature - dp.perp2();
      theValid = (dist_tmp > 0.);
      theCenter = p1 + dp + ec * sqrt(std::abs(dist_tmp));
    }
    bool isValid() const { return theValid; }
    const Point& center() const { return theCenter; }
    const long double& curvature() const { return theCurvature; }

  private:
    bool theValid;
    long double theCurvature;
    Point theCenter;
  };

public:
  L1MuonPixelTrackFitter(const edm::ParameterSet& cfg);

  virtual ~L1MuonPixelTrackFitter() {}

  void setL1Constraint(const L1MuGMTCand& muon);
  void setPxConstraint(const SeedingHitSet& hits);

  virtual reco::Track* run(const MagneticField& field,
                           const std::vector<const TrackingRecHit*>& hits,
                           const TrackingRegion& region) const;

  static double getBending(double invPt, double eta, int charge);
  static double getBendingError(double invPt, double eta);

private:
  double valInversePt(double phi0, double phiL1, double eta) const;
  double errInversePt(double invPt, double eta) const;

  double valPhi(const Circle& c, int charge) const;
  double errPhi(double invPt, double eta) const;

  double valCotTheta(const PixelRecoLineRZ& line) const;
  double errCotTheta(double invPt, double eta) const;

  double valZip(double curvature, const GlobalPoint& p0, const GlobalPoint& p1) const;
  double errZip(double invPt, double eta) const;

  double valTip(const Circle& c, double curvature) const;
  double errTip(double invPt, double eta) const;

  double findPt(double phi0, double phiL1, double eta, int charge) const;
  double deltaPhi(double phi1, double phi2) const;
  static void param(double eta, double& p1, double& p2, double& p3);

private:
  edm::ParameterSet theConfig;

  const double invPtErrorScale;
  const double phiErrorScale;
  const double cotThetaErrorScale;
  const double tipErrorScale;
  const double zipErrorScale;

  // L1 constraint
  double thePhiL1, theEtaL1;
  int theChargeL1;

  // Px constraint
  GlobalPoint theHit1, theHit2;

private:
  friend class L1Seeding;
};
#endif
