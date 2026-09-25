#ifndef HcalHitMaker_h
#define HcalHitMaker_h

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloHitMakers/interface/EcalHitMaker.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Math/Transform3D.h"

class CaloGeometryHelper;

class HcalHitMaker : public CaloHitMaker {
public:
  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef ROOT::Math::Transform3D Transform3D;

  HcalHitMaker(EcalHitMaker&, unsigned);
  ~HcalHitMaker() override { ; }

  /// Set the spot energy
  inline void setSpotEnergy(double e) override { spotEnergy = e; }

  /// add the hit in the HCAL in local coordinates
  bool addHit(double r, double phi, unsigned layer = 0) override;

  /// add the hit in the HCAL in global coordinates
  bool addHit(const XYZPoint& point, unsigned layer = 0);

  // get the hits
  const CaloHitMap& getHits() override { return hitMap_; }

  /// set the depth in X0 or Lambda0 units depending on showerType
  bool setDepth(double, bool inCm = false);

private:
  EcalHitMaker& myGrid;

  const FSimTrack* myTrack;
  XYZPoint ecalEntrance_;
  XYZVector particleDirection;
  int onHcal;

  double currentDepth_;
  Transform3D locToGlobal_;
  double radiusFactor_;
  bool mapCalculated_;

public:
  static int getSubHcalDet(const FSimTrack* t) {
    if (t->onHcal() == 1)
      return HcalBarrel;
    if (t->onHcal() == 2)
      return HcalEndcap;
    if (t->onVFcal() == 2)
      return HcalForward;
    return -1;
  }
};
#endif
