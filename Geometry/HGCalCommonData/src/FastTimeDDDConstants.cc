#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

FastTimeDDDConstants::FastTimeDDDConstants(const FastTimeParameters* ft) : ftpar_(ft) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeDDDConstants::FastTimeDDDConstants "
                                << "( const FastTimeParameters* ft ) constructor";
#endif
  initialize();
}

FastTimeDDDConstants::~FastTimeDDDConstants() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeDDDConstants:destructed!!!";
#endif
}

std::pair<int, int> FastTimeDDDConstants::getZPhi(double z, double phi) const {
  if (phi < 0)
    phi += (2 * geant_units::piRadians);
  int iz = (int)(z / dZBarrel_) + 1;
  if (iz > ftpar_->nZBarrel_)
    iz = ftpar_->nZBarrel_;
  int iphi = (int)(phi / dPhiBarrel_) + 1;
  if (iphi > ftpar_->nPhiBarrel_)
    iphi = 1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeDDDConstants:Barrel z|phi " << z << " " << convertRadToDeg(phi)
                                << " iz|iphi " << iz << " " << iphi;
#endif
  return std::pair<int, int>(iz, iphi);
}

std::pair<int, int> FastTimeDDDConstants::getEtaPhi(double r, double phi) const {
  if (phi < 0)
    phi += (2 * geant_units::piRadians);
  int ir(ftpar_->nEtaEndcap_);
  for (unsigned int k = 1; k < rLimits_.size(); ++k) {
    if (r > rLimits_[k]) {
      ir = k;
      break;
    }
  }
  int iphi = (int)(phi / dPhiEndcap_) + 1;
  if (iphi > ftpar_->nPhiEndcap_)
    iphi = 1;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeDDDConstants:Endcap r|phi " << r << " " << convertRadToDeg(phi)
                                << " ir|iphi " << ir << " " << iphi;
#endif
  return std::pair<int, int>(ir, iphi);
}

GlobalPoint FastTimeDDDConstants::getPosition(int type, int izeta, int iphi, int zside) const {
  double x(0), y(0), z(0);
  if (type == 1) {
    double phi = (iphi - 0.5) * dPhiBarrel_;
    x = ftpar_->geomParBarrel_[2] * cos(phi);
    y = ftpar_->geomParBarrel_[2] * sin(phi);
    z = ftpar_->geomParBarrel_[0] + (izeta - 0.5) * dZBarrel_;
  } else if (type == 2) {
    double phi = (iphi - 0.5) * dPhiEndcap_;
    double r = (izeta <= 0 || izeta >= (int)(rLimits_.size())) ? 0 : 0.5 * (rLimits_[izeta - 1] + rLimits_[izeta]);
    x = (zside < 0) ? -r * cos(phi) : r * cos(phi);
    y = r * sin(phi);
    z = ftpar_->geomParEndcap_[2];
  }
  if (zside < 0)
    z = -z;
  GlobalPoint p(x, y, z);
  return p;
}

std::vector<GlobalPoint> FastTimeDDDConstants::getCorners(int type, int izeta, int iphi, int zside) const {
  double x(0), y(0), z(0), dx(0), dz(0), r(0), phi(0);
  if (type == 1) {
    phi = (iphi - 0.5) * dPhiBarrel_;
    r = ftpar_->geomParBarrel_[2];
    x = r * cos(phi);
    y = r * sin(phi);
    z = ftpar_->geomParBarrel_[0] + (izeta - 0.5) * dZBarrel_;
    dx = 0.5 * ftpar_->geomParBarrel_[3];
    dz = 0.5 * dZBarrel_;
  } else if (type == 2) {
    phi = (iphi - 0.5) * dPhiEndcap_;
    r = (izeta <= 0 || izeta >= (int)(rLimits_.size())) ? 0 : 0.5 * (rLimits_[izeta - 1] + rLimits_[izeta]);
    x = (zside < 0) ? -r * cos(phi) : r * cos(phi);
    y = r * sin(phi);
    z = ftpar_->geomParEndcap_[2];
    dx = 0.5 * r * dPhiEndcap_;
    dz = 0.5 * ftpar_->geomParEndcap_[3];
  }
  if (zside < 0) {
    z = -z;
    dz = -dz;
  }
  static const int signx[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
  static const int signy[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
  static const int signz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
  std::vector<GlobalPoint> pts;
  for (unsigned int i = 0; i != 8; ++i) {
    GlobalPoint p(x + signx[i] * dx, y + signy[i] * dx, z + signz[i] * dz);
    pts.emplace_back(p);
  }
  return pts;
}

int FastTimeDDDConstants::getCells(int type) const {
  int numb(0);
  if (type == 1) {
    numb = (ftpar_->nZBarrel_) * (ftpar_->nPhiBarrel_);
  } else if (type == 2) {
    numb = (ftpar_->nEtaEndcap_) * (ftpar_->nPhiEndcap_);
  }
  return numb;
}

double FastTimeDDDConstants::getRin(int type) const {
  double value(0);
  if (type == 1) {
    value = (ftpar_->geomParBarrel_[2]);
  } else if (type == 2) {
    value = (ftpar_->geomParEndcap_[0]);
  }
  return value;
}

double FastTimeDDDConstants::getRout(int type) const {
  double value(0);
  if (type == 1) {
    value = (ftpar_->geomParBarrel_[2] + ftpar_->geomParBarrel_[3]);
  } else if (type == 2) {
    value = (ftpar_->geomParEndcap_[1]);
  }
  return value;
}

double FastTimeDDDConstants::getZHalf(int type) const {
  double value(0);
  if (type == 1) {
    value = 0.5 * (ftpar_->geomParBarrel_[1] - ftpar_->geomParBarrel_[0]);
  } else if (type == 2) {
    value = (ftpar_->geomParEndcap_[3]);
  }
  return value;
}

double FastTimeDDDConstants::getZPos(int type) const {
  double value(0);
  if (type == 1) {
    value = 0.5 * (ftpar_->geomParBarrel_[1] + ftpar_->geomParBarrel_[0]);
  } else if (type == 2) {
    value = (ftpar_->geomParEndcap_[2]);
  }
  return value;
}

bool FastTimeDDDConstants::isValidXY(int type, int izeta, int iphi) const {
  bool ok(false);
  if (type == 1) {
    ok = ((izeta > 0) && (izeta <= ftpar_->nZBarrel_) && (iphi > 0) && (iphi <= ftpar_->nPhiBarrel_));
  } else if (type == 2) {
    ok = ((izeta > 0) && (izeta <= ftpar_->nEtaEndcap_) && (iphi > 0) && (iphi <= ftpar_->nPhiEndcap_));
  }
  return ok;
}

int FastTimeDDDConstants::numberEtaZ(int type) const {
  int numb(0);
  if (type == 1) {
    numb = (ftpar_->nZBarrel_);
  } else if (type == 2) {
    numb = (ftpar_->nEtaEndcap_);
  }
  return numb;
}

int FastTimeDDDConstants::numberPhi(int type) const {
  int numb(0);
  if (type == 1) {
    numb = (ftpar_->nPhiBarrel_);
  } else if (type == 2) {
    numb = (ftpar_->nPhiEndcap_);
  }
  return numb;
}

void FastTimeDDDConstants::initialize() {
  double thmin = atan(ftpar_->geomParEndcap_[0] / ftpar_->geomParEndcap_[2]);
  etaMax_ = -log(0.5 * thmin);
  double thmax = atan(ftpar_->geomParEndcap_[1] / ftpar_->geomParEndcap_[2]);
  etaMin_ = -log(0.5 * thmax);
  dEta_ = (etaMax_ - etaMin_) / ftpar_->nEtaEndcap_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Theta range " << convertRadToDeg(thmin) << ":" << convertRadToDeg(thmax)
                                << " Eta range " << etaMin_ << ":" << etaMax_ << ":" << dEta_;
#endif
  for (int k = 0; k <= ftpar_->nEtaEndcap_; ++k) {
    double eta = etaMin_ + k * dEta_;
    double theta = 2.0 * atan(exp(-eta));
    double rval = (ftpar_->geomParEndcap_[2]) * tan(theta);
    rLimits_.emplace_back(rval);
  }
  dZBarrel_ = ftpar_->geomParBarrel_[1] / ftpar_->nZBarrel_;
  dPhiBarrel_ = (2 * geant_units::piRadians) / ftpar_->nPhiBarrel_;
  dPhiEndcap_ = (2 * geant_units::piRadians) / ftpar_->nPhiEndcap_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeDDDConstants initialized with " << ftpar_->nZBarrel_ << ":"
                                << ftpar_->nPhiBarrel_ << ":" << getCells(1) << " cells for barrel; dz|dphi "
                                << dZBarrel_ << "|" << dPhiBarrel_ << " and " << ftpar_->nEtaEndcap_ << ":"
                                << ftpar_->nPhiEndcap_ << ":" << getCells(2) << " cells for endcap; dphi "
                                << dPhiEndcap_ << " The Limits in R are";
  for (unsigned int k = 0; k < rLimits_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << rLimits_[k] << " ";
#endif
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(FastTimeDDDConstants);
