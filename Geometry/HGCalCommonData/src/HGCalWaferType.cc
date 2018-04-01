#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<cmath>

//#define EDM_ML_DEBUG

const double k_ScaleFromDDD = 0.1;

HGCalWaferType::HGCalWaferType(const std::vector<double>& rad100, 
			       const std::vector<double>& rad200,
			       double waferSize, double zMin, 
			       int cutValue) : rad100_(rad100),
					       rad200_(rad200),
					       waferSize_(waferSize),
					       zMin_(zMin),
					       cutValue_(cutValue) {
  r_ = 0.5*waferSize_;
  R_ = sqrt3_*waferSize_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferType: initialized with waferR's "
				<< waferSize_ << ":" << r_ << ":" << R_
				<< " Cut " << cutValue_ << " zMin " << zMin_
				<< " with " << rad100_.size() << ":"
				<< rad200_.size() << " parameters for R:";
  for (unsigned k=0; k<rad100_.size(); ++k) 
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] 100:200 " << rad100_[k]
				  << " 200:300 " << rad200_[k];
#endif
}

HGCalWaferType::~HGCalWaferType() { }

int HGCalWaferType::getType(double xpos, double ypos, double zpos) {
  double xc[6], yc[6];
  xc[0] = xpos+r_;  yc[0] = ypos+0.5*R_;
  xc[1] = xpos;     yc[1] = ypos+R_;
  xc[2] = xpos-r_;  yc[2] = ypos+0.5*R_;
  xc[3] = xpos-r_;  yc[3] = ypos-0.5*R_;
  xc[4] = xpos;     yc[4] = ypos-R_;
  xc[5] = xpos+r_;  yc[5] = ypos-0.5*R_;
  std::pair<double,double> rv = rLimits(zpos);
  int fine(0), coarse(0);
  for (int k=0; k<6; ++k) {
    double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
    if      (rpos <= rv.first)  ++fine;
    else if (rpos <= rv.second) ++coarse;
  }
  int type = 2;
  if      (fine   >= cutValue_) type = 0;
  else if (coarse >= cutValue_) type = 1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferType: position " << xpos << ":"
				<< ypos << ":" << zpos << " R " << ":" 
				<< rv.first << ":" << rv.second
				<< " corners|type " << fine << ":" << coarse 
				<< ":" << type ;
#endif
  return type;
}

std::pair<double,double> HGCalWaferType::rLimits(double zpos) {
  double zz = std::abs(zpos);
  if (zz < zMin_) zz = zMin_;
  zz *= k_ScaleFromDDD;
  double rfine   = rad100_[0];
  double rcoarse = rad200_[0];
  for (int i=1; i<5; ++i) {
    rfine   *= zz; rfine   += rad100_[i];
    rcoarse *= zz; rcoarse += rad200_[i];
  }
  return std::pair<double,double>(rfine/k_ScaleFromDDD,rcoarse/k_ScaleFromDDD);
}

