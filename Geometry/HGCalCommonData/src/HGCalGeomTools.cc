#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"

#include <string>

//#define EDM_ML_DEBUG

void HGCalGeomTools::radius(double zf, double zb,
			    std::vector<double> const& zFront1,
			    std::vector<double> const& rFront1,
			    std::vector<double> const& slope1,
			    std::vector<double> const& zFront2,
			    std::vector<double> const& rFront2,
			    std::vector<double> const& slope2, int flag,
			    std::vector<double>& zz, 
			    std::vector<double>& rin,
			    std::vector<double>& rout) {

  double dz1(0), dz2(0);
  auto zf1 = std::lower_bound(zFront1.begin(), zFront1.end(), zf);
  if (zf1 != zFront1.begin()) --zf1;
  if (std::abs(*(zf1+1) - zf) < tol) { ++zf1; dz1 = 2*tol;}
  auto zf2 = std::lower_bound(zFront2.begin(), zFront2.end(), zf);
  if (zf2 != zFront2.begin()) --zf2;
  auto zb1 = std::lower_bound(zFront1.begin(), zFront1.end(), zb);
  if (zb1 != zFront1.begin()) --zb1;
  if (std::abs(*zb1 - zb) < tol) {--zb1; dz2 =-2*tol;}
  auto zb2 = std::lower_bound(zFront2.begin(), zFront2.end(), zb);
  if (zb2 != zFront2.begin()) --zb2;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "HGCalGeomTools::radius:zf " << zf << " : " << *zf1 << " : " << *zf2 
    << " zb " << zb << " : " << *zb1 << " : " << *zb2 << " Flag " << flag;
#endif
  if ((zf1==zb1) && (zf2==zb2)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomTools::radius:Try " << zf << ":"
				  << zb << " dz " << dz1 << ":" << dz2;
#endif
    zz.emplace_back(zf);
    rin.emplace_back(radius(zf+dz1,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zf,zFront2,rFront2,slope2));
    zz.emplace_back(zb);
    rin.emplace_back(radius(zb+dz2,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zb,zFront2,rFront2,slope2));
  } else if (zf1 == zb1) {
#ifdef EDM_ML_DEBUG
    double z1 = std::max(*zf2,*zb2);
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomTools::radius:Try " << zf << ":"
				  << *zb2 << " (" << z1 << ") : " << zb;
#endif
    zz.emplace_back(zf);
    rin.emplace_back(radius(zf,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zf,zFront2,rFront2,slope2));
    if (slope(*zb2,zFront2,slope2) < tol) {
      zz.emplace_back(*zb2);
      rin.emplace_back(radius(*zb2,zFront1,rFront1,slope1));
      rout.emplace_back(radius(*zb2-tol,zFront2,rFront2,slope2));
    }
    zz.emplace_back(*zb2);
    rin.emplace_back(radius(*zb2,zFront1,rFront1,slope1));
    rout.emplace_back(radius(*zb2,zFront2,rFront2,slope2));
    zz.emplace_back(zb);
    rin.emplace_back(radius(zb,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zb,zFront2,rFront2,slope2));
  } else if (zf2 == zb2) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomTools::radius:Try " << zf << ":"
				  << *zb1 << ":" << zb;
#endif
    zz.emplace_back(zf);
    rin.emplace_back(radius(zf,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zf,zFront2,rFront2,slope2));
    if (slope(*zb1,zFront1,slope1) < tol) {
      zz.emplace_back(*zb1);
      rin.emplace_back(radius(*zb1-tol,zFront1,rFront1,slope1));
      rout.emplace_back(radius(*zb1,zFront2,rFront2,slope2));
    }
    zz.emplace_back(*zb1);
    rin.emplace_back(radius(*zb1,zFront1,rFront1,slope1));
    rout.emplace_back(radius(*zb1,zFront2,rFront2,slope2));
    zz.emplace_back(zb);
    rin.emplace_back(radius(zb,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zb,zFront2,rFront2,slope2));
  } else {
    double z1 = std::min(*zf2,*zb1);
    double z2 = std::max(*zf2,*zb1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomTools::radius:Try " << zf << ":"
				  << z1 << " : " << z2 << ":" << zb;
#endif
    zz.emplace_back(zf);
    rin.emplace_back(radius(zf,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zf,zFront2,rFront2,slope2));
    zz.emplace_back(z1);
    rin.emplace_back(radius(z1,zFront1,rFront1,slope1));
    rout.emplace_back(radius(z1,zFront2,rFront2,slope2));
    zz.emplace_back(z2);
    rin.emplace_back(radius(z2,zFront1,rFront1,slope1));
    rout.emplace_back(radius(z2,zFront2,rFront2,slope2));
    zz.emplace_back(zb);
    rin.emplace_back(radius(zb,zFront1,rFront1,slope1));
    rout.emplace_back(radius(zb,zFront2,rFront2,slope2));
  }
  double rmin = *(std::min_element(rout.begin(),rout.end()));
  if (flag > 1) {
    for (auto& rr : rout) rr = rmin;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") 
    << "HGCalGeomTools::radius has " << zz.size() << " sections: "<< rmin;
  for (unsigned int k=0; k<zz.size(); ++k) 
    edm::LogVerbatim("HGCalGeom") 
      << "[" << k << "] Z = " << zz[k] << " R = " << rin[k] << ":" << rout[k];
#endif
}

double HGCalGeomTools::radius(double z, std::vector<double> const& zFront,
                              std::vector<double> const& rFront,
                              std::vector<double> const& slope) {

  auto itrz = std::lower_bound(zFront.begin(), zFront.end(), z);
  if (itrz != zFront.begin()) --itrz;
  unsigned int ik = static_cast<unsigned int>(itrz-zFront.begin());
  if (ik < zFront.size() && std::abs(z-zFront[ik+1]) < tol) ++ik;
  double r = rFront[ik] + (z - zFront[ik]) * slope[ik];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
      << "DDHGCalGeomTools: Z " << z << " k " << ik << " R " << r;
#endif
  return r;
}

double HGCalGeomTools::radius(double z, int layer0, int layerf,
                              std::vector<double> const& zFront,
                              std::vector<double> const& rFront) {
  double r = rFront[0];
#ifdef EDM_ML_DEBUG
  unsigned int ik(0);
#endif
  for (unsigned int k = 0; k < rFront.size(); ++k) {
    int k1 = layerf - layer0 + (int)(k);
    if (k1 < (int)(zFront.size())) {
      r = rFront[k];
#ifdef EDM_ML_DEBUG
      ik = k;
#endif
      if (z < zFront[k1] + tol) break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
      << "DDHGCalGeomTools: Z " << z << ":" << ik << " R " << r;
#endif
  return r;
}

double HGCalGeomTools::slope(double z, std::vector<double> const& zFront,
			     std::vector<double> const& slope) {

  auto itrz = std::lower_bound(zFront.begin(), zFront.end(), z);
  if (itrz != zFront.begin()) --itrz;
  unsigned int ik = static_cast<unsigned int>(itrz-zFront.begin());
  //  if (ik < zFront.size() && std::abs(z-zFront[ik+1]) < tol) ++ik;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomTools::slope:z " << z << " k " 
				<< ik << " Slope " << slope[ik];
#endif
  return slope[ik];
}

std::pair<double, double> HGCalGeomTools::zradius(
    double z1, double z2, std::vector<double> const& zF,
    std::vector<double> const& rF) {
  double z(-1), r(-1);
  for (unsigned int k = 0; k < rF.size(); ++k) {
    if ((z1 > zF[k] - tol) && (z2 < zF[k] + tol)) {
      z = zF[k];
      r = rF[k];
      break;
    }
  }
  return std::make_pair(z, r);
}

std::pair<int32_t, int32_t> HGCalGeomTools::waferCorner(double xpos,
                                                        double ypos, double r,
                                                        double R, double rMin,
                                                        double rMax,
                                                        bool oldBug) {
  double xc[HGCalParameters::k_CornerSize], yc[HGCalParameters::k_CornerSize];
  xc[0] = xpos;
  yc[0] = ypos + R;
  xc[1] = xpos - r;
  yc[1] = ypos + 0.5 * R;
  if (oldBug) {
    xc[2] = xpos + r;
    yc[2] = ypos - 0.5 * R;
  } else {
    xc[2] = xpos - r;
    yc[2] = ypos - 0.5 * R;
  }
  xc[3] = xpos;
  yc[3] = ypos - R;
  xc[4] = xpos + r;
  yc[4] = ypos - 0.5 * R;
  xc[5] = xpos + r;
  yc[5] = ypos + 0.5 * R;
  int32_t nCorner(0), firstCorner(-1), firstMiss(-1);
#ifdef EDM_ML_DEBUG
  std::vector<uint32_t> corners;
#endif
  for (uint32_t k = 0; k < HGCalParameters::k_CornerSize; ++k) {
    double rpos = sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
    if ((rpos <= rMax) && (rpos >= rMin)) {
#ifdef EDM_ML_DEBUG
      corners.emplace_back(k);
#endif
      if (firstCorner < 0) firstCorner = k;
      ++nCorner;
    } else {
      if (firstMiss < 0) firstMiss = k;
    }
  }
  if ((nCorner > 1) && (firstCorner == 0) && (firstMiss < nCorner)) {
    firstCorner = firstMiss + HGCalParameters::k_CornerSize - nCorner;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom")
      << "waferCorner:: R " << rMin << ":" << rMax << nCorner
      << " corners; first corner " << firstCorner;
  for (uint32_t k = 0; k < HGCalParameters::k_CornerSize; ++k) {
    double rpos = std::sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
    std::string ok =
        (std::find(corners.begin(), corners.end(), k) != corners.end())
            ? " In"
            : " Out";
    edm::LogVerbatim("HGCalGeom") << "Corner[" << k << "] x " << xc[k] << " y "
                                  << yc[k] << " R " << rpos << ok;
  }
#endif
  return std::make_pair(nCorner, firstCorner);
}
