#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
#include "DataFormats/Math/interface/angle_units.h"
#include <sstream>

#define EDM_ML_DEBUG

void HGCalCassette::setParameter(int cassette, const std::vector<double>& shifts) {
  cassette_ = cassette;
  typeHE_ = (cassette_ >= 12);
  shifts_.insert(shifts_.end(), shifts.begin(), shifts.end());
  double dphi = angle_units::piRadians / cassette_;
  for (int k = 0; k < cassette_; ++k) {
    double angle = (2 * k - 1) * dphi;
    cos_.emplace_back(std::cos(angle));
    sin_.emplace_back(std::sin(angle));
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "# of cassettes = " << cassette_ << " Type " << typeHE_ << " dPhi " << dphi;
  std::ostringstream st1;
  st1 << " Shifts:";
  for (const auto& s : shifts_)
    st1 << ":" << s;
  edm::LogVerbatim("HGCalGeom") << st1.str();
  std::ostringstream st2;
  st2 << " Cos|Sin:";
  for (int k = 0; k < cassette_; ++k)
    st2 << "  " << cos_[k] << ":" << sin_[k];
  edm::LogVerbatim("HGCalGeom") << st2.str();
#endif
}

std::pair<double, double> HGCalCassette::getShift(int layer, int zside, int cassette) {
  int locc = (zside < 0) ? (cassette - 1) : (typeHE_ ? positHE_[cassette - 1] : positEE_[cassette - 1]);
  int loc = cassette_ * (layer - 1) + locc;
  std::pair<double, double> xy = std::make_pair(shifts_[loc] * cos_[locc], shifts_[loc] * sin_[locc]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalCassette::getShift: Layer " << layer << " zside " << zside << " cassette "
                                << cassette << " Loc " << locc << ":" << loc << " shift " << xy.first << ":"
                                << xy.second;
#endif
  return xy;
}
