#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
