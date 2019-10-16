#include "Geometry/HGCalGeometry/interface/HGCalMouseBite.h"
#include "DataFormats/Math/interface/CMSUnits.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

HGCalMouseBite::HGCalMouseBite(const HGCalDDDConstants& hgc, const bool rot) {
  bool modeUV =
      ((hgc.geomMode() == HGCalGeometryMode::Hexagon8) || (hgc.geomMode() == HGCalGeometryMode::Hexagon8Full));
  if (modeUV) {
    const std::vector<double> angle = {90._deg, 30._deg};
    std::vector<std::pair<double, double> > projXY;
    for (auto ang : angle)
      projXY.emplace_back(std::make_pair(cos(ang), sin(ang)));
    const double mousebite(hgc.mouseBite(true));
    const double wafersize(hgc.waferSize(true));
    double cut = wafersize * tan(angle[1]) - mousebite;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Creating HGCMosueBite with cut at " << cut << " along " << angle.size()
                                  << " axes in wafers of size " << wafersize;
    for (unsigned int k = 0; k < angle.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "Axis[" << k << "] " << convertRadToDeg(angle[k]) << " with projections "
                                    << projXY[k].first << ":" << projXY[k].second;
#endif
    static const double sqrt3 = std::sqrt(3.0);
    int nf(HGCSiliconDetId::HGCalFineN);
    int nf2 = nf / 2;
    double Rf = wafersize / (3.0 * nf);
    double rf = 0.5 * Rf * sqrt3;
    for (int u = 0; u < 2 * nf; ++u) {
      for (int v = 0; v < 2 * nf; ++v) {
        if (((v - u) < nf) && ((u - v) <= nf)) {
          double yp = std::abs((u - 0.5 * v - nf2) * 2 * rf);
          double xp = std::abs((1.5 * (v - nf) + 1.0) * Rf);
          for (auto proj : projXY) {
            double dist = (rot ? (yp * proj.first + xp * proj.second) : (xp * proj.first + yp * proj.second));
            if (dist > cut) {
              rejectFine_.emplace_back(100 * u + v);
              break;
            }
          }
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalMouseBite:: " << rejectFine_.size()
                                  << " masked u, v's among the fine grain wafers:";
    for (unsigned int k = 0; k < rejectFine_.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] = (" << rejectFine_[k] / 100 << ", " << rejectFine_[k] % 100
                                    << ")";
#endif
    int nc(HGCSiliconDetId::HGCalCoarseN);
    int nc2 = nc / 2;
    double Rc = hgc.getParameter()->waferSize_ / (3.0 * nc);
    double rc = 0.5 * Rc * sqrt3;
    for (int u = 0; u < 2 * nc; ++u) {
      for (int v = 0; v < 2 * nc; ++v) {
        if (((v - u) < nc) && ((u - v) <= nc)) {
          double yp = (u - 0.5 * v - nc2) * 2 * rc;
          double xp = (1.5 * (v - nc) + 1.0) * Rc;
          for (auto proj : projXY) {
            double dist = (rot ? (yp * proj.first + xp * proj.second) : (xp * proj.first + yp * proj.second));
            if (dist > cut)
              rejectCoarse_.emplace_back(100 * u + v);
            break;
          }
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "HGCalMouseBite:: " << rejectCoarse_.size()
                                  << " masked u, v's among the coarse grain wafers:";
    for (unsigned int k = 0; k < rejectCoarse_.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] = (" << rejectCoarse_[k] / 100 << ", " << rejectCoarse_[k] % 100
                                    << ")";
#endif
  }
}
