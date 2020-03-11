#include "Geometry/HcalCommonData/interface/HcalGeomParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

static const double tan10deg = std::tan(10._deg);

void HcalGeomParameters::getConstRHO(std::vector<double>& rHO) const {
  rHO.emplace_back(rminHO_);
  for (double i : etaHO_)
    rHO.emplace_back(i);
}

std::vector<int> HcalGeomParameters::getModHalfHBHE(const int type) const {
  std::vector<int> modHalf;
  if (type == 0) {
    modHalf.emplace_back(nmodHB_);
    modHalf.emplace_back(nzHB_);
  } else {
    modHalf.emplace_back(nmodHE_);
    modHalf.emplace_back(nzHE_);
  }
  return modHalf;
}

void HcalGeomParameters::loadGeometry(const DDFilteredView& _fv, HcalParameters& php) {
  DDFilteredView fv = _fv;
  bool dodet = true;
  bool hf = false;
  clear(php);

  while (dodet) {
    DDTranslation t = fv.translation();
    std::vector<int> copy = fv.copyNumbers();
    const DDSolid& sol = fv.logicalPart().solid();
    int idet = 0, lay = -1;
    int nsiz = static_cast<int>(copy.size());
    if (nsiz > 0)
      lay = copy[nsiz - 1] / 10;
    if (nsiz > 1)
      idet = copy[nsiz - 2] / 1000;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Name " << fv.logicalPart().solid().name() << " Copy " << copy.size();
#endif
    double dx = 0, dy = 0, dz = 0, dx1 = 0, dx2 = 0;
    double alp(0);
    if (sol.shape() == DDSolidShape::ddbox) {
      const DDBox& box = static_cast<DDBox>(sol);
      dx = HcalGeomParameters::k_ScaleFromDDDToG4 * box.halfX();
      dy = HcalGeomParameters::k_ScaleFromDDDToG4 * box.halfY();
      dz = HcalGeomParameters::k_ScaleFromDDDToG4 * box.halfZ();
    } else if (sol.shape() == DDSolidShape::ddtrap) {
      const DDTrap& trp = static_cast<DDTrap>(sol);
      dx1 = HcalGeomParameters::k_ScaleFromDDDToG4 * trp.x1();
      dx2 = HcalGeomParameters::k_ScaleFromDDDToG4 * trp.x2();
      dx = 0.25 * HcalGeomParameters::k_ScaleFromDDDToG4 * (trp.x1() + trp.x2() + trp.x3() + trp.x4());
      dy = 0.5 * HcalGeomParameters::k_ScaleFromDDDToG4 * (trp.y1() + trp.y2());
      dz = HcalGeomParameters::k_ScaleFromDDDToG4 * trp.halfZ();
      alp = 0.5 * (trp.alpha1() + trp.alpha2());
    } else if (sol.shape() == DDSolidShape::ddtubs) {
      const DDTubs& tub = static_cast<DDTubs>(sol);
      dx = HcalGeomParameters::k_ScaleFromDDDToG4 * tub.rIn();
      dy = HcalGeomParameters::k_ScaleFromDDDToG4 * tub.rOut();
      dz = HcalGeomParameters::k_ScaleFromDDDToG4 * tub.zhalf();
    }
    if (idet == 3) {
      // HB
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "HB " << sol.name() << " Shape " << sol.shape() << " Layer " << lay << " R "
                                   << t.Rho();
#endif
      if (lay >= 0 && lay < maxLayer_) {
        ib_[lay]++;
        rb_[lay] += (HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho());
        if (thkb_[lay] <= 0) {
          if (lay < 17)
            thkb_[lay] = dx;
          else
            thkb_[lay] = std::min(dx, dy);
        }
        if (lay < 17) {
          bool found = false;
          for (double k : rxb_) {
            if (std::abs(k - (HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho())) < 0.01) {
              found = true;
              break;
            }
          }
          if (!found) {
            rxb_.emplace_back(HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho());
            php.rhoxHB.emplace_back(HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho() * std::cos(t.phi()));
            php.zxHB.emplace_back(HcalGeomParameters::k_ScaleFromDDDToG4 * std::abs(t.z()));
            php.dyHB.emplace_back(2. * dy);
            php.dxHB.emplace_back(2. * dz);
            php.layHB.emplace_back(lay);
          }
        }
      }
      if (lay == 2) {
        int iz = copy[nsiz - 5];
        int fi = copy[nsiz - 4];
        unsigned int it1 = find(iz, izb_);
        if (it1 == izb_.size())
          izb_.emplace_back(iz);
        unsigned int it2 = find(fi, phib_);
        if (it2 == phib_.size())
          phib_.emplace_back(fi);
      }
      if (lay == 18) {
        int ifi = -1, ich = -1;
        if (nsiz > 2)
          ifi = copy[nsiz - 3];
        if (nsiz > 3)
          ich = copy[nsiz - 4];
        double z1 = std::abs((HcalGeomParameters::k_ScaleFromDDDToG4 * t.z()) + dz);
        double z2 = std::abs((HcalGeomParameters::k_ScaleFromDDDToG4 * t.z()) - dz);
        if (std::abs(z1 - z2) < 0.01)
          z1 = 0;
        if (ifi == 1 && ich == 4) {
          if (z1 > z2) {
            double tmp = z1;
            z1 = z2;
            z2 = tmp;
          }
          bool sok = true;
          for (unsigned int kk = 0; kk < php.zHO.size(); kk++) {
            if (std::abs(z2 - php.zHO[kk]) < 0.01) {
              sok = false;
              break;
            } else if (z2 < php.zHO[kk]) {
              php.zHO.resize(php.zHO.size() + 2);
              for (unsigned int kz = php.zHO.size() - 1; kz > kk + 1; kz = kz - 2) {
                php.zHO[kz] = php.zHO[kz - 2];
                php.zHO[kz - 1] = php.zHO[kz - 3];
              }
              php.zHO[kk + 1] = z2;
              php.zHO[kk] = z1;
              sok = false;
              break;
            }
          }
          if (sok) {
            php.zHO.emplace_back(z1);
            php.zHO.emplace_back(z2);
          }
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "Detector " << idet << " Lay " << lay << " fi " << ifi << " " << ich << " z "
                                       << z1 << " " << z2;
#endif
        }
      }
    } else if (idet == 4) {
      // HE
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "HE " << sol.name() << " Shape " << sol.shape() << " Layer " << lay << " Z "
                                   << t.z();
#endif
      if (lay >= 0 && lay < maxLayer_) {
        ie_[lay]++;
        ze_[lay] += std::abs(HcalGeomParameters::k_ScaleFromDDDToG4 * t.z());
        if (thke_[lay] <= 0)
          thke_[lay] = dz;
        double rinHE = HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho() * cos(alp) - dy;
        double routHE = HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho() * cos(alp) + dy;
        rminHE_[lay] += rinHE;
        rmaxHE_[lay] += routHE;
        bool found = false;
        for (double k : php.zxHE) {
          if (std::abs(k - std::abs(HcalGeomParameters::k_ScaleFromDDDToG4 * t.z())) < 0.01) {
            found = true;
            break;
          }
        }
        if (!found) {
          php.zxHE.emplace_back(HcalGeomParameters::k_ScaleFromDDDToG4 * std::abs(t.z()));
          php.rhoxHE.emplace_back(HcalGeomParameters::k_ScaleFromDDDToG4 * t.Rho() * std::cos(t.phi()));
          php.dyHE.emplace_back(dy * std::cos(t.phi()));
          dx1 -= 0.5 * (HcalGeomParameters::k_ScaleFromDDDToG4 * t.rho() - dy) * std::cos(t.phi()) * tan10deg;
          dx2 -= 0.5 * (HcalGeomParameters::k_ScaleFromDDDToG4 * t.rho() + dy) * std::cos(t.phi()) * tan10deg;
          php.dx1HE.emplace_back(-dx1);
          php.dx2HE.emplace_back(-dx2);
          php.layHE.emplace_back(lay);
        }
      }
      if (copy[nsiz - 1] == kHELayer1_ || copy[nsiz - 1] == kHELayer2_) {
        int iz = copy[nsiz - 7];
        int fi = copy[nsiz - 5];
        unsigned int it1 = find(iz, ize_);
        if (it1 == ize_.size())
          ize_.emplace_back(iz);
        unsigned int it2 = find(fi, phie_);
        if (it2 == phie_.size())
          phie_.emplace_back(fi);
      }
    } else if (idet == 5) {
      // HF
      if (!hf) {
        const std::vector<double>& paras = sol.parameters();
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "HF " << sol.name() << " Shape " << sol.shape() << " Z " << t.z() << " with "
                                     << paras.size() << " Parameters";
        for (unsigned j = 0; j < paras.size(); j++)
          edm::LogVerbatim("HCalGeom") << "HF Parameter[" << j << "] = " << paras[j];
#endif
        if (sol.shape() == DDSolidShape::ddpolycone_rrz) {
          int nz = (int)(paras.size()) - 3;
          dzVcal_ = 0.5 * HcalGeomParameters::k_ScaleFromDDDToG4 * (paras[nz] - paras[3]);
          hf = true;
        } else if (sol.shape() == DDSolidShape::ddtubs || sol.shape() == DDSolidShape::ddcons) {
          dzVcal_ = HcalGeomParameters::k_ScaleFromDDDToG4 * paras[0];
          hf = true;
        }
      }
#ifdef EDM_ML_DEBUG
    } else {
      edm::LogVerbatim("HCalGeom") << "Unknown Detector " << idet << " for " << sol.name() << " Shape " << sol.shape()
                                   << " R " << t.Rho() << " Z " << t.z();
#endif
    }
    dodet = fv.next();
  }

  loadfinal(php);
}

void HcalGeomParameters::loadGeometry(const cms::DDCompactView* cpv, HcalParameters& php) {
  cms::DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());
  std::string attribute = "OnlyForHcalSimNumbering";
  cms::DDSpecParRefs ref;
  const cms::DDSpecParRegistry& mypar = cpv->specpars();
  mypar.filter(ref, attribute, "HCAL");
  fv.mergedSpecifics(ref);
  clear(php);
  bool hf(false);
  while (fv.firstChild()) {
    auto t = fv.translation();
    std::vector<double> paras = fv.parameters();
    std::vector<int> copy = fv.copyNos();
    int idet = 0, lay = -1;
    int nsiz = static_cast<int>(copy.size());
    if (nsiz > 0)
      lay = copy[0] / 10;
    if (nsiz > 1)
      idet = copy[1] / 1000;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Name " << fv.name() << " Copy " << copy.size();
    for (unsigned int n = 0; n < copy.size(); ++n)
      edm::LogVerbatim("HCalGeom") << "[" << n << "] " << copy[n];
    edm::LogVerbatim("HCalGeom") << "Detector " << idet << " Layer " << lay << " parameters: " << paras.size();
    for (unsigned int n = 0; n < paras.size(); ++n)
      edm::LogVerbatim("HCalGeom") << "[" << n << "] " << paras[n];
#endif
    double dx = 0, dy = 0, dz = 0, dx1 = 0, dx2 = 0;
    double alp(0);
    if (fv.isABox()) {
      dx = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[0];
      dy = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[1];
      dz = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[2];
    } else if (fv.isATrapezoid()) {
      dx1 = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[4];
      dx2 = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[5];
      dx = 0.25 * HcalGeomParameters::k_ScaleFromDD4HepToG4 * (paras[4] + paras[5] + paras[8] + paras[9]);
      dy = 0.5 * HcalGeomParameters::k_ScaleFromDD4HepToG4 * (paras[3] + paras[7]);
      dz = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[0];
      alp = 0.5 * (paras[6] + paras[10]);
    } else if (fv.isATubeSeg()) {
      dx = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[0];
      dy = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[1];
      dz = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[2];
    }
    if (idet == 3) {
      // HB
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "HB " << fv.name() << " Shape " << cms::dd::name(cms::DDSolidShapeMap, fv.shape())
                                   << " Layer " << lay << " R " << t.Rho();
#endif
      if (lay >= 0 && lay < maxLayer_) {
        ib_[lay]++;
        rb_[lay] += (HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho());
        if (thkb_[lay] <= 0) {
          if (lay < 17)
            thkb_[lay] = dx;
          else
            thkb_[lay] = std::min(dx, dy);
        }
        if (lay < 17) {
          bool found = false;
          for (double k : rxb_) {
            if (std::abs(k - (HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho())) < 0.01) {
              found = true;
              break;
            }
          }
          if (!found) {
            rxb_.emplace_back(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho());
            php.rhoxHB.emplace_back(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho() * std::cos(t.phi()));
            php.zxHB.emplace_back(std::abs(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z()));
            php.dyHB.emplace_back(2. * dy);
            php.dxHB.emplace_back(2. * dz);
            php.layHB.emplace_back(lay);
          }
        }
      }
      if (lay == 2) {
        int iz = copy[4];
        int fi = copy[3];
        unsigned int it1 = find(iz, izb_);
        if (it1 == izb_.size())
          izb_.emplace_back(iz);
        unsigned int it2 = find(fi, phib_);
        if (it2 == phib_.size())
          phib_.emplace_back(fi);
      }
      if (lay == 18) {
        int ifi = -1, ich = -1;
        if (nsiz > 2)
          ifi = copy[2];
        if (nsiz > 3)
          ich = copy[3];
        double z1 = std::abs((HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z()) + dz);
        double z2 = std::abs((HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z()) - dz);
        if (std::abs(z1 - z2) < 0.01)
          z1 = 0;
        if (ifi == 1 && ich == 4) {
          if (z1 > z2) {
            double tmp = z1;
            z1 = z2;
            z2 = tmp;
          }
          bool sok = true;
          for (unsigned int kk = 0; kk < php.zHO.size(); kk++) {
            if (std::abs(z2 - php.zHO[kk]) < 0.01) {
              sok = false;
              break;
            } else if (z2 < php.zHO[kk]) {
              php.zHO.resize(php.zHO.size() + 2);
              for (unsigned int kz = php.zHO.size() - 1; kz > kk + 1; kz = kz - 2) {
                php.zHO[kz] = php.zHO[kz - 2];
                php.zHO[kz - 1] = php.zHO[kz - 3];
              }
              php.zHO[kk + 1] = z2;
              php.zHO[kk] = z1;
              sok = false;
              break;
            }
          }
          if (sok) {
            php.zHO.emplace_back(z1);
            php.zHO.emplace_back(z2);
          }
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HCalGeom") << "Detector " << idet << " Lay " << lay << " fi " << ifi << " " << ich << " z "
                                       << z1 << " " << z2;
#endif
        }
      }
    } else if (idet == 4) {
      // HE
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "HE " << fv.name() << " Shape " << cms::dd::name(cms::DDSolidShapeMap, fv.shape())
                                   << " Layer " << lay << " Z " << t.z();
#endif
      if (lay >= 0 && lay < maxLayer_) {
        ie_[lay]++;
        ze_[lay] += std::abs(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z());
        if (thke_[lay] <= 0)
          thke_[lay] = dz;
        double rinHE = HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho() * cos(alp) - dy;
        double routHE = HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho() * cos(alp) + dy;
        rminHE_[lay] += rinHE;
        rmaxHE_[lay] += routHE;
        bool found = false;
        for (double k : php.zxHE) {
          if (std::abs(k - std::abs(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z())) < 0.01) {
            found = true;
            break;
          }
        }
        if (!found) {
          php.zxHE.emplace_back(std::abs(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z()));
          php.rhoxHE.emplace_back(HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho() * std::cos(t.phi()));
          php.dyHE.emplace_back(dy * std::cos(t.phi()));
          dx1 -= 0.5 * (HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho() - dy) * std::cos(t.phi()) * tan10deg;
          dx2 -= 0.5 * (HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho() + dy) * std::cos(t.phi()) * tan10deg;
          php.dx1HE.emplace_back(-dx1);
          php.dx2HE.emplace_back(-dx2);
          php.layHE.emplace_back(lay);
        }
      }
      if (copy[0] == kHELayer1_ || copy[0] == kHELayer2_) {
        int iz = copy[6];
        int fi = copy[4];
        unsigned int it1 = find(iz, ize_);
        if (it1 == ize_.size())
          ize_.emplace_back(iz);
        unsigned int it2 = find(fi, phie_);
        if (it2 == phie_.size())
          phie_.emplace_back(fi);
      }
    } else if (idet == 5) {
      // HF
      if (!hf) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "HF " << fv.name() << " Shape "
                                     << cms::dd::name(cms::DDSolidShapeMap, fv.shape()) << " Z "
                                     << HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z() << " with " << paras.size()
                                     << " Parameters";
        for (unsigned j = 0; j < paras.size(); j++)
          edm::LogVerbatim("HCalGeom") << "HF Parameter[" << j << "] = " << paras[j];
#endif
        if (fv.isA<dd4hep::Polycone>()) {
          int nz = (int)(paras.size()) - 3;
          dzVcal_ = 0.5 * HcalGeomParameters::k_ScaleFromDD4HepToG4 * (paras[nz] - paras[3]);
          hf = true;
        } else if (fv.isATubeSeg() || fv.isAConeSeg()) {
          dzVcal_ = HcalGeomParameters::k_ScaleFromDD4HepToG4 * paras[2];
          hf = true;
        }
      }
#ifdef EDM_ML_DEBUG
    } else {
      edm::LogVerbatim("HCalGeom") << "Unknown Detector " << idet << " for " << fv.name() << " Shape "
                                   << cms::dd::name(cms::DDSolidShapeMap, fv.shape()) << " R "
                                   << (HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.Rho()) << " Z "
                                   << (HcalGeomParameters::k_ScaleFromDD4HepToG4 * t.z());
#endif
    }
  }
  loadfinal(php);
}

unsigned int HcalGeomParameters::find(int element, std::vector<int>& array) const {
  unsigned int id = array.size();
  for (unsigned int i = 0; i < array.size(); i++) {
    if (element == array[i]) {
      id = i;
      break;
    }
  }
  return id;
}

double HcalGeomParameters::getEta(double r, double z) const {
  double tmp = 0;
  if (z != 0)
    tmp = -log(tan(0.5 * atan(r / z)));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalGeomParameters::getEta " << r << " " << z << " ==> " << tmp;
#endif
  return tmp;
}

void HcalGeomParameters::clear(HcalParameters& php) {
  php.rhoxHB.clear();
  php.zxHB.clear();
  php.dyHB.clear();
  php.dxHB.clear();
  php.layHB.clear();
  php.layHE.clear();
  php.zxHE.clear();
  php.rhoxHE.clear();
  php.dyHE.clear();
  php.dx1HE.clear();
  php.dx2HE.clear();

  // Initialize all variables
  nzHB_ = nmodHB_ = 0;
  nzHE_ = nmodHE_ = 0;
  for (int i = 0; i < 4; ++i)
    etaHO_[i] = 0;
  zVcal_ = dzVcal_ = dlShort_ = 0;
  rminHO_ = dzVcal_ = -1.;
  for (int i = 0; i < maxLayer_; ++i) {
    rb_.emplace_back(0.0);
    ze_.emplace_back(0.0);
    thkb_.emplace_back(-1.0);
    thke_.emplace_back(-1.0);
    ib_.emplace_back(0);
    ie_.emplace_back(0);
    rminHE_.emplace_back(0.0);
    rmaxHE_.emplace_back(0.0);
  }
}

void HcalGeomParameters::loadfinal(HcalParameters& php) {
  int ibmx = 0, iemx = 0;
  for (int i = 0; i < maxLayer_; i++) {
    if (ib_[i] > 0) {
      rb_[i] /= static_cast<double>(ib_[i]);
      ibmx = i + 1;
    }
    if (ie_[i] > 0) {
      ze_[i] /= static_cast<double>(ie_[i]);
      rminHE_[i] /= static_cast<double>(ie_[i]);
      rmaxHE_[i] /= static_cast<double>(ie_[i]);
      iemx = i + 1;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Index " << i << " Barrel " << ib_[i] << " " << rb_[i] << " Endcap " << ie_[i]
                                 << " " << ze_[i] << ":" << rminHE_[i] << ":" << rmaxHE_[i];
#endif
  }
  for (int i = 4; i >= 0; i--) {
    if (ib_[i] == 0) {
      rb_[i] = rb_[i + 1];
      thkb_[i] = thkb_[i + 1];
    }
    if (ie_[i] == 0) {
      ze_[i] = ze_[i + 1];
      thke_[i] = thke_[i + 1];
    }
#ifdef EDM_ML_DEBUG
    if (ib_[i] == 0 || ie_[i] == 0)
      edm::LogVerbatim("HCalGeom") << "Index " << i << " Barrel " << ib_[i] << " " << rb_[i] << " Endcap " << ie_[i]
                                   << " " << ze_[i];
#endif
  }

#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.layHB.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "HB: " << php.layHB[k] << " R " << rxb_[k] << " " << php.rhoxHB[k] << " Z "
                                 << php.zxHB[k] << " DY " << php.dyHB[k] << " DZ " << php.dxHB[k];
  for (unsigned int k = 0; k < php.layHE.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "HE: " << php.layHE[k] << " R " << php.rhoxHE[k] << " Z " << php.zxHE[k]
                                 << " X1|X2 " << php.dx1HE[k] << "|" << php.dx2HE[k] << " DY " << php.dyHE[k];
  edm::LogVerbatim("HCalGeom") << "HcalGeomParameters: Maximum Layer for HB " << ibmx << " for HE " << iemx
                               << " extent " << dzVcal_;
#endif

  if (ibmx > 0) {
    php.rHB.resize(ibmx);
    php.drHB.resize(ibmx);
    for (int i = 0; i < ibmx; i++) {
      php.rHB[i] = rb_[i];
      php.drHB[i] = thkb_[i];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "HcalGeomParameters: php.rHB[" << i << "] = " << php.rHB[i] << " php.drHB[" << i
                                   << "] = " << php.drHB[i];
#endif
    }
  }
  if (iemx > 0) {
    php.zHE.resize(iemx);
    php.dzHE.resize(iemx);
    for (int i = 0; i < iemx; i++) {
      php.zHE[i] = ze_[i];
      php.dzHE[i] = thke_[i];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "HcalGeomParameters: php.zHE[" << i << "] = " << php.zHE[i] << " php.dzHE[" << i
                                   << "] = " << php.dzHE[i];
#endif
    }
  }

  nzHB_ = static_cast<int>(izb_.size());
  nmodHB_ = static_cast<int>(phib_.size());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalGeomParameters::loadGeometry: " << nzHB_ << " barrel half-sectors";
  for (int i = 0; i < nzHB_; i++)
    edm::LogVerbatim("HCalGeom") << "Section " << i << " Copy number " << izb_[i];
  edm::LogVerbatim("HCalGeom") << "HcalGeomParameters::loadGeometry: " << nmodHB_ << " barrel modules";
  for (int i = 0; i < nmodHB_; i++)
    edm::LogVerbatim("HCalGeom") << "Module " << i << " Copy number " << phib_[i];
#endif

  nzHE_ = static_cast<int>(ize_.size());
  nmodHE_ = static_cast<int>(phie_.size());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalGeomParameters::loadGeometry: " << nzHE_ << " endcap half-sectors";
  for (int i = 0; i < nzHE_; i++)
    edm::LogVerbatim("HCalGeom") << "Section " << i << " Copy number " << ize_[i];
  edm::LogVerbatim("HCalGeom") << "HcalGeomParameters::loadGeometry: " << nmodHE_ << " endcap modules";
  for (int i = 0; i < nmodHE_; i++)
    edm::LogVerbatim("HCalGeom") << "Module " << i << " Copy number " << phie_[i];
#endif

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HO has Z of size " << php.zHO.size();
  for (unsigned int kk = 0; kk < php.zHO.size(); kk++)
    edm::LogVerbatim("HCalGeom") << "ZHO[" << kk << "] = " << php.zHO[kk];
#endif
  if (ibmx > 17 && php.zHO.size() > 4) {
    rminHO_ = php.rHB[17] - 100.0;
    etaHO_[0] = getEta(0.5 * (php.rHB[17] + php.rHB[18]), php.zHO[1]);
    etaHO_[1] = getEta(php.rHB[18] + php.drHB[18], php.zHO[2]);
    etaHO_[2] = getEta(php.rHB[18] - php.drHB[18], php.zHO[3]);
    etaHO_[3] = getEta(php.rHB[18] + php.drHB[18], php.zHO[4]);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HO Eta boundaries " << etaHO_[0] << " " << etaHO_[1] << " " << etaHO_[2] << " "
                               << etaHO_[3];
  edm::LogVerbatim("HCalGeom") << "HO Parameters " << rminHO_ << " " << php.zHO.size();
  for (unsigned int i = 0; i < php.zHO.size(); ++i)
    edm::LogVerbatim("HCalGeom") << " zho[" << i << "] = " << php.zHO[i];
#endif
}
