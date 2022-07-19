#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

//#define EDM_ML_DEBUG

HGCalWaferType::HGCalWaferType(const std::vector<double>& rad100,
                               const std::vector<double>& rad200,
                               double waferSize,
                               double zMin,
                               int choice,
                               unsigned int cornerCut,
                               double cutArea)
    : rad100_(rad100),
      rad200_(rad200),
      waferSize_(waferSize),
      zMin_(zMin),
      choice_(choice),
      cutValue_(cornerCut),
      cutFracArea_(cutArea) {
  r_ = 0.5 * waferSize_;
  R_ = sqrt3_ * waferSize_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferType: initialized with waferR's " << waferSize_ << ":" << r_ << ":" << R_
                                << " Choice " << choice_ << " Cuts " << cutValue_ << ":" << cutFracArea_ << " zMin "
                                << zMin_ << " with " << rad100_.size() << ":" << rad200_.size() << " parameters for R:";
  for (unsigned k = 0; k < rad100_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] 100:200 " << rad100_[k] << " 200:300 " << rad200_[k];
#endif
}

int HGCalWaferType::getCassette(int index, const HGCalParameters::waferInfo_map& wafers) {
  auto itr = wafers.find(index);
  return ((itr == wafers.end()) ? -1 : ((itr->second).cassette));
}

int HGCalWaferType::getOrient(int index, const HGCalParameters::waferInfo_map& wafers) {
  auto itr = wafers.find(index);
  return ((itr == wafers.end()) ? -1 : ((itr->second).orient));
}

int HGCalWaferType::getPartial(int index, const HGCalParameters::waferInfo_map& wafers) {
  auto itr = wafers.find(index);
  return ((itr == wafers.end()) ? -1 : ((itr->second).part));
}

int HGCalWaferType::getType(int index, const HGCalParameters::waferInfo_map& wafers) {
  auto itr = wafers.find(index);
  return ((itr == wafers.end()) ? -1 : ((itr->second).type));
}

int HGCalWaferType::getType(int index, const std::vector<int>& indices, const std::vector<int>& properties) {
  auto itr = std::find(std::begin(indices), std::end(indices), index);
  int type = (itr == std::end(indices))
                 ? -1
                 : HGCalProperty::waferThick(properties[static_cast<unsigned int>(itr - std::begin(indices))]);
  return type;
}

int HGCalWaferType::getType(double xpos, double ypos, double zpos) {
  std::vector<double> xc(HGCalParameters::k_CornerSize, 0);
  std::vector<double> yc(HGCalParameters::k_CornerSize, 0);
  xc[0] = xpos + r_;
  yc[0] = ypos + 0.5 * R_;
  xc[1] = xpos;
  yc[1] = ypos + R_;
  xc[2] = xpos - r_;
  yc[2] = ypos + 0.5 * R_;
  xc[3] = xpos - r_;
  yc[3] = ypos - 0.5 * R_;
  xc[4] = xpos;
  yc[4] = ypos - R_;
  xc[5] = xpos + r_;
  yc[5] = ypos - 0.5 * R_;
  const auto& rv = rLimits(zpos);
  std::vector<int> fine, coarse;
  for (unsigned int k = 0; k < HGCalParameters::k_CornerSize; ++k) {
    double rpos = std::sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
    if (rpos <= rv.first)
      fine.emplace_back(k);
    else if (rpos <= rv.second)
      coarse.emplace_back(k);
  }
  int type(-2);
  double fracArea(0);
  if (choice_ == 1) {
    if (fine.size() >= cutValue_)
      type = HGCSiliconDetId::HGCalFine;
    else if (coarse.size() >= cutValue_)
      type = HGCSiliconDetId::HGCalCoarseThin;
    else
      type = HGCSiliconDetId::HGCalCoarseThick;
  } else {
    if (fine.size() >= 4)
      type = HGCSiliconDetId::HGCalFine;
    else if (coarse.size() >= 4 && fine.size() <= 1)
      type = HGCSiliconDetId::HGCalCoarseThin;
    else if (coarse.size() < 2 && fine.empty())
      type = HGCSiliconDetId::HGCalCoarseThick;
    else if (!fine.empty())
      type = -1;
    if (type <= -1) {
      unsigned int kmax = (type == -1) ? fine.size() : coarse.size();
      std::vector<double> xcn, ycn;
      for (unsigned int k = 0; k < kmax; ++k) {
        unsigned int k1 = (type == -1) ? fine[k] : coarse[k];
        unsigned int k2 = (k1 == xc.size() - 1) ? 0 : k1 + 1;
        bool ok = ((type == -1) ? (std::find(fine.begin(), fine.end(), k2) != fine.end())
                                : (std::find(coarse.begin(), coarse.end(), k2) != coarse.end()));
        xcn.emplace_back(xc[k1]);
        ycn.emplace_back(yc[k1]);
        if (!ok) {
          double rr = (type == -1) ? rv.first : rv.second;
          const auto& xy = intersection(k1, k2, xc, yc, xpos, ypos, rr);
          xcn.emplace_back(xy.first);
          ycn.emplace_back(xy.second);
        }
      }
      fracArea = areaPolygon(xcn, ycn) / areaPolygon(xc, yc);
      type = (fracArea > cutFracArea_) ? -(1 + type) : -type;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferType: position " << xpos << ":" << ypos << ":" << zpos << " R "
                                << ":" << rv.first << ":" << rv.second << " corners|type " << fine.size() << ":"
                                << coarse.size() << ":" << fracArea << ":" << type;
#endif
  return type;
}

std::pair<double, double> HGCalWaferType::rLimits(double zpos) {
  double zz = std::abs(zpos);
  if (zz < zMin_)
    zz = zMin_;
  zz *= HGCalParameters::k_ScaleFromDDD;
  double rfine = rad100_[0];
  double rcoarse = rad200_[0];
  for (int i = 1; i < 5; ++i) {
    rfine *= zz;
    rfine += rad100_[i];
    rcoarse *= zz;
    rcoarse += rad200_[i];
  }
  rfine *= HGCalParameters::k_ScaleToDDD;
  rcoarse *= HGCalParameters::k_ScaleToDDD;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferType: Z " << zpos << ":" << zz << " R " << rfine << ":" << rcoarse;
#endif
  return std::make_pair(rfine, rcoarse);
}

double HGCalWaferType::areaPolygon(std::vector<double> const& x, std::vector<double> const& y) {
  double area = 0.0;
  int n = static_cast<int>(x.size());
  int j = n - 1;
  for (int i = 0; i < n; ++i) {
    area += ((x[j] + x[i]) * (y[i] - y[j]));
    j = i;
  }
  return (0.5 * area);
}

std::pair<double, double> HGCalWaferType::intersection(
    int k1, int k2, std::vector<double> const& x, std::vector<double> const& y, double xpos, double ypos, double rr) {
  double slope = (x[k1] - x[k2]) / (y[k1] - y[k2]);
  double interc = x[k1] - slope * y[k1];
  double xx[2], yy[2], dist[2];
  double v1 = std::sqrt((slope * slope + 1) * rr * rr - (interc * interc));
  yy[0] = (-slope * interc + v1) / (1 + slope * slope);
  yy[1] = (-slope * interc - v1) / (1 + slope * slope);
  for (int i = 0; i < 2; ++i) {
    xx[i] = (slope * yy[i] + interc);
    dist[i] = ((xx[i] - xpos) * (xx[i] - xpos)) + ((yy[i] - ypos) * (yy[i] - ypos));
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferType: InterSection " << dist[0] << ":" << xx[0] << ":" << yy[0] << " vs "
                                << dist[1] << ":" << xx[1] << ":" << yy[1];
#endif
  if (dist[0] > dist[1])
    return std::make_pair(xx[1], yy[1]);
  else
    return std::make_pair(xx[0], yy[0]);
}
