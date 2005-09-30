#include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
#include <math.h>

namespace calogeom {

  IdealObliquePrism::IdealObliquePrism(const GlobalPoint& faceCenter, float widthEta, float widthPhi, float thickness, bool parallelToZaxis) : 
  cms::CaloCellGeometry(faceCenter),
  hwidthEta_(widthEta/2),
  hwidthPhi_(widthPhi/2),
  thickness_((parallelToZaxis)?(thickness):(-thickness))
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
  
  IdealObliquePrism::IdealObliquePrism(float eta, float phi, float radialDistanceToFront, float widthEta, float widthPhi, float thickness, bool parallelToZaxis) :
    cms::CaloCellGeometry(etaPhiR(eta,phi,radialDistanceToFront)),
    hwidthEta_(widthEta/2),
    hwidthPhi_(widthPhi/2),
  thickness_((parallelToZaxis)?(thickness):(-thickness))
  {
  }

  const std::vector<GlobalPoint> & IdealObliquePrism::getCorners() const {
    if (points_.empty()) {
      if (thickness_>0) { 
	/* In this case, the faces are parallel to the zaxis.  This implies that all corners will have the same 
	   cylindrical radius. 
	*/
	GlobalPoint p=getPosition();
	float r_near=p.perp()/cos(hwidthPhi_);
	float r_far=r_near*((p.mag()+thickness_)/(p.mag())); // geometry...
	float eta=p.eta();
	float phi=p.phi();
	points_.push_back(etaPhiPerp(eta+hwidthEta_,phi+hwidthPhi_,r_near)); // (+,+,near)
	points_.push_back(etaPhiPerp(eta+hwidthEta_,phi-hwidthPhi_,r_near)); // (+,-,near)
	points_.push_back(etaPhiPerp(eta-hwidthEta_,phi+hwidthPhi_,r_near)); // (-,+,near)
	points_.push_back(etaPhiPerp(eta-hwidthEta_,phi-hwidthPhi_,r_near)); // (-,-,near)
	points_.push_back(etaPhiPerp(eta+hwidthEta_,phi+hwidthPhi_,r_far)); // (+,+,far)
	points_.push_back(etaPhiPerp(eta+hwidthEta_,phi-hwidthPhi_,r_far)); // (+,-,far)
	points_.push_back(etaPhiPerp(eta-hwidthEta_,phi+hwidthPhi_,r_far)); // (-,+,far)
	points_.push_back(etaPhiPerp(eta-hwidthEta_,phi-hwidthPhi_,r_far)); // (-,-,far)
      } else {
	/* In this case, the faces are perpendicular to the zaxis.  This implies that all corners will have the same 
	   z-dimension. 
	*/
	GlobalPoint p=getPosition();
	float z_near=p.z();
	float z_far=z_near*((p.mag()-thickness_)/(p.mag())); // geometry... (negative to correct sign)
	float eta=p.eta();
	float phi=p.phi();
	points_.push_back(etaPhiZ(eta+hwidthEta_,phi+hwidthPhi_,z_near)); // (+,+,near)
	points_.push_back(etaPhiZ(eta+hwidthEta_,phi-hwidthPhi_,z_near)); // (+,-,near)
	points_.push_back(etaPhiZ(eta-hwidthEta_,phi+hwidthPhi_,z_near)); // (-,+,near)
	points_.push_back(etaPhiZ(eta-hwidthEta_,phi-hwidthPhi_,z_near)); // (-,-,near)
	points_.push_back(etaPhiZ(eta+hwidthEta_,phi+hwidthPhi_,z_far)); // (+,+,far)
	points_.push_back(etaPhiZ(eta+hwidthEta_,phi-hwidthPhi_,z_far)); // (+,-,far)
	points_.push_back(etaPhiZ(eta-hwidthEta_,phi+hwidthPhi_,z_far)); // (-,+,far)
	points_.push_back(etaPhiZ(eta-hwidthEta_,phi-hwidthPhi_,z_far)); // (-,-,far)
      }    
    }
    return points_;
  }

  bool IdealObliquePrism::inside(const GlobalPoint& point) const {
    // first check eta/phi
    bool is_inside=true;
    const GlobalPoint& face=getPosition();
    // eta
    is_inside=is_inside && fabs(point.eta()-face.eta())<=hwidthEta_;
    // phi
    is_inside=is_inside && fabs(point.phi()-face.phi())<=hwidthPhi_;

    // distance (trickier)
    if (is_inside) {
      GlobalPoint face2=etaPhiR(face.eta(),face.phi(),face.mag()+fabs(thickness_));
      if (thickness_>0) { // 
	float projection=point.perp()*cos(point.phi()-face.phi());
	is_inside=is_inside && projection>=face.perp();
	is_inside=is_inside && projection<=face2.perp();
      } else { // here, it is just a Z test.
	is_inside=is_inside && ((face.z()<0)?(point.z()<=face.z()):(point.z()>=face.z())); // "front" face
	is_inside=is_inside && ((face.z()<0)?(point.z()>=face2.z()):(point.z()<=face2.z())); // "back" face
      }
    }
    return is_inside;
  }
}
