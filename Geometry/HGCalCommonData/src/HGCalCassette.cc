#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalCassette.h"
#include <algorithm>
#include <sstream>

//#define EDM_ML_DEBUG

void HGCalCassette::setParameter(int cassette, const std::vector<double>& shifts) {
  cassette_ = cassette;
  typeHE_ = (cassette_ >= 12);
  shifts_.insert(shifts_.end(), shifts.begin(), shifts.end());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "# of cassettes = " << cassette_ << " Type " << typeHE_;
  for (uint32_t j1 = 0; j1 < shifts.size(); j1 += 12) {
    std::ostringstream st1;
    if (j1 == 0)
      st1 << " Shifts:";
    else
      st1 << "        ";
    uint32_t j2 = std::min((j1 + 12), static_cast<uint32_t>(shifts.size()));
    for (uint32_t j = j1; j < j2; ++j)
      st1 << ":" << shifts[j];
    edm::LogVerbatim("HGCalGeom") << st1.str();
  }
#endif
}

std::pair<double, double> HGCalCassette::getShift(int layer, int zside, int cassette) const {
  int locc = (zside < 0) ? (cassette - 1) : (typeHE_ ? positHE_[cassette - 1] : positEE_[cassette - 1]);
  int loc = 2 * (cassette_ * (layer - 1) + locc);
  std::pair<double, double> xy = std::make_pair(-zside * shifts_[loc], shifts_[loc + 1]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalCassette::getShift: Layer " << layer << " zside " << zside << " type "
                                << typeHE_ << " cassette " << cassette << " Loc " << locc << ":" << loc << " shift "
                                << xy.first << ":" << xy.second;
#endif
  return xy;
}

int HGCalCassette::cassetteIndex(int det, int layer, int side, int cassette) {
  int zs = (side > 0) ? factor_ : 0;
  return (((zs + det) * factor_ + layer) * factor_ + cassette);
}

int HGCalCassette::cassetteType(int det, int zside, int cassette) {
  int type = (zside < 0) ? cassette : ((det == 0) ? (1 + positEE_[cassette - 1]) : (1 + positHE_[cassette - 1]));
  return type;
}
