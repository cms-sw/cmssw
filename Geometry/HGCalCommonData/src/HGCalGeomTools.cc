#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>

//#define EDM_ML_DEBUG

double HGCalGeomTools::radius(double z, std::vector<double> const& zFront,
			      std::vector<double> const& rFront,
			      std::vector<double> const& slope) {

  double r(0);
#ifdef EDM_ML_DEBUG
  unsigned int ik(0);
#endif
  for (unsigned int k=0; k<slope.size(); ++k) {
    if (z < zFront[k]) break;
    r  = rFront[k] + (z - zFront[k]) * slope[k];
#ifdef EDM_ML_DEBUG
    ik = k;
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalGeomTools: Z " << z << ":" << ik 
				<< " R " << r;
#endif
  return r;
}

double HGCalGeomTools::radius(double z, int layer0, int layerf,
			      std::vector<double> const& zFront,
			      std::vector<double> const& rFront) {

  double r(0);
#ifdef EDM_ML_DEBUG
  unsigned int ik(0);
#endif
  for (unsigned int k=0; k<rFront.size(); ++k) {
    int k1 = layerf-layer0+(int)(k);
    if (k1 < (int)(zFront.size())) {
      if (z < zFront[k1]) break;
      r  = rFront[k];
#ifdef EDM_ML_DEBUG
      ik = k;
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalGeomTools: Z " << z << ":" << ik 
				<< " R " << r;
#endif
  return r;
}

std::pair<int32_t,int32_t> HGCalGeomTools::waferCorner(double xpos, 
						       double ypos,
						       double r, double R, 
						       double rMin, 
						       double rMax,
						       bool oldBug) {
  double xc[HGCalParameters::k_CornerSize], yc[HGCalParameters::k_CornerSize];
  xc[0] = xpos;    yc[0] = ypos+R;
  xc[1] = xpos-r;  yc[1] = ypos+0.5*R;
  if (oldBug) {
    xc[2] = xpos+r;  yc[2] = ypos-0.5*R;
  } else {
    xc[2] = xpos-r;  yc[2] = ypos-0.5*R;
  }
  xc[3] = xpos;    yc[3] = ypos-R;
  xc[4] = xpos+r;  yc[4] = ypos-0.5*R;
  xc[5] = xpos+r;  yc[5] = ypos+0.5*R;
  int32_t  nCorner(0), firstCorner(-1), firstMiss(-1);
#ifdef EDM_ML_DEBUG
  std::vector<uint32_t> corners;
#endif
  for (uint32_t k=0; k<HGCalParameters::k_CornerSize; ++k) {
    double rpos = sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
    if ((rpos <= rMax) && (rpos >= rMin)) {
#ifdef EDM_ML_DEBUG
      corners.emplace_back(k);
#endif
      if (firstCorner < 0) firstCorner = k;
      ++nCorner;
    } else {
      if (firstMiss < 0)   firstMiss = k;
    }
  }
  if ((nCorner > 1) && (firstCorner == 0) && (firstMiss < nCorner)) {
    firstCorner = firstMiss+HGCalParameters::k_CornerSize-nCorner;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "waferCorner:: R " << rMin << ":" << rMax
				<< nCorner << " corners; first corner "
				<< firstCorner;
  for (uint32_t k=0; k<HGCalParameters::k_CornerSize; ++k) {
    double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
    std::string ok = (std::find(corners.begin(),corners.end(),k) != 
		      corners.end()) ? " In" : " Out";
    edm::LogVerbatim("HGCalGeom") << "Corner[" << k << "] x " << xc[k] 
				  << " y " << yc[k] << " R " << rpos << ok;
  }
#endif
  return std::make_pair(nCorner,firstCorner);
}
