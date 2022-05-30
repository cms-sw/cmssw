#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
#include <sstream>

//#define EDM_ML_DEBUG

void HGCalCassette::setParameter(int cassette, const std::vector<double>& shifts) {
  cassette_ = cassette;
  typeHE_ = (cassette_ >= 12);
  shifts_.insert(shifts_.end(), shifts.begin(), shifts.end());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "# of cassettes = " << cassette_ << " Type " << typeHE_;
  std::ostringstream st1;
  st1 << " Shifts:";
  for (const auto& s : shifts_)
    st1 << ":" << s;
  edm::LogVerbatim("HGCalGeom") << st1.str();
#endif
}

std::pair<double, double> HGCalCassette::getShift(int layer, int zside, int cassette) const {
  int locc = (zside < 0) ? (cassette - 1) : (typeHE_ ? positHE_[cassette - 1] : positEE_[cassette - 1]);
  int loc = 2 * (cassette_ * (layer - 1) + locc);
  std::pair<double, double> xy = std::make_pair(shifts_[loc], shifts_[loc + 1]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalCassette::getShift: Layer " << layer << " zside " << zside << " cassette "
                                << cassette << " Loc " << locc << ":" << loc << " shift " << xy.first << ":"
                                << xy.second;
#endif
  return xy;
}
