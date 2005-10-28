#include "Geometry/CaloGeometry/interface/IdealZPrism.h"
#include <math.h>

namespace calogeom {

  IdealZPrism::IdealZPrism(const GlobalPoint& faceCenter, float widthEta, float widthPhi, float deltaZ) : 
  CaloCellGeometry(faceCenter),
  hwidthEta_(widthEta/2),
  hwidthPhi_(widthPhi/2),
  deltaZ_(deltaZ)
{
}

  static inline GlobalPoint etaPhiR(float eta, float phi, float rad) {
    return GlobalPoint(rad*cosf(phi)/coshf(eta),rad*sinf(phi)/coshf(eta),rad*tanhf(eta));
  }

  static inline GlobalPoint etaPhiPerp(float eta, float phi, float perp) {
    return GlobalPoint(perp*cosf(phi),perp*sinf(phi),perp*sinhf(eta));
  }

  static inline GlobalPoint etaPhiZ(float eta, float phi, float z) {
    return GlobalPoint(z*cosf(phi)/sinhf(eta),z*sinf(phi)/sinhf(eta),z);
  }
  
  IdealZPrism::IdealZPrism(float eta, float phi, float radialDistanceToFront, float widthEta, float widthPhi, float deltaZ) :
    CaloCellGeometry(etaPhiR(eta,phi,radialDistanceToFront)),
    hwidthEta_(widthEta/2),
    hwidthPhi_(widthPhi/2),
  deltaZ_(deltaZ)
  {
  }

  const std::vector<GlobalPoint> & IdealZPrism::getCorners() const {
    if (points_.empty()) {
      GlobalPoint p=getPosition();
      float z_near=p.z();
      float z_far=z_near+deltaZ_*p.z()/fabs(p.z());
      float eta=p.eta();
      float phi=p.phi();
      points_.push_back(etaPhiZ(eta+hwidthEta_,phi+hwidthPhi_,z_near)); // (+,+,near)
      points_.push_back(etaPhiZ(eta+hwidthEta_,phi-hwidthPhi_,z_near)); // (+,-,near)
      points_.push_back(etaPhiZ(eta-hwidthEta_,phi-hwidthPhi_,z_near)); // (-,-,near)
      points_.push_back(etaPhiZ(eta-hwidthEta_,phi+hwidthPhi_,z_near)); // (-,+,near)
      points_.push_back(GlobalPoint(points_[0].x(),points_[0].y(),z_far)); // (+,+,far)
      points_.push_back(GlobalPoint(points_[1].x(),points_[1].y(),z_far)); // (+,-,far)
      points_.push_back(GlobalPoint(points_[2].x(),points_[2].y(),z_far)); // (-,-,far)
      points_.push_back(GlobalPoint(points_[3].x(),points_[3].y(),z_far)); // (-,+,far)	
    }
    return points_;
  }

  bool IdealZPrism::inside(const GlobalPoint& point) const {
    // first check eta/phi
    bool is_inside=true;
    const GlobalPoint& face=getPosition();
    // eta
    is_inside=is_inside && fabs(point.eta()-face.eta())<=hwidthEta_;
    // phi
    is_inside=is_inside && fabs(point.phi()-face.phi())<=hwidthPhi_;

    // distance 
    if (point.z()<0) {
      is_inside=is_inside && (point.z()<=face.z()) && (point.z()>(face.z()-deltaZ_));
    } else {
      is_inside=is_inside && (point.z()>=face.z()) && (point.z()<(face.z()+deltaZ_));
    }

    return is_inside;
  }
}
