#include "Geometry/HGCalTBCommonData/interface/HGCalTBGeomParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

#define EDM_ML_DEBUG
using namespace geant_units::operators;

const double tolerance = 0.001;
const double tolmin = 1.e-20;

HGCalTBGeomParameters::HGCalTBGeomParameters() : sqrt3_(std::sqrt(3.0)) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters::HGCalTBGeomParameters "
                                << "constructor";
#endif
}

void HGCalTBGeomParameters::loadGeometryHexagon(const DDFilteredView& _fv,
                                                HGCalTBParameters& php,
                                                const std::string& sdTag1,
                                                const DDCompactView* cpv,
                                                const std::string& sdTag2,
                                                const std::string& sdTag3,
                                                HGCalGeometryMode::WaferMode mode) {
  DDFilteredView fv = _fv;
  bool dodet(true);
  std::map<int, HGCalTBGeomParameters::layerParameters> layers;
  std::vector<HGCalTBParameters::hgtrform> trforms;
  std::vector<bool> trformUse;

  while (dodet) {
    const DDSolid& sol = fv.logicalPart().solid();
    // Layers first
    std::vector<int> copy = fv.copyNumbers();
    int nsiz = static_cast<int>(copy.size());
    int lay = (nsiz > 0) ? copy[nsiz - 1] : 0;
    int zp = (nsiz > 2) ? copy[nsiz - 3] : -1;
    if (zp != 1)
      zp = -1;
    if (lay == 0) {
      throw cms::Exception("DDException") << "Funny layer # " << lay << " zp " << zp << " in " << nsiz << " components";
    } else {
      if (std::find(php.layer_.begin(), php.layer_.end(), lay) == php.layer_.end())
        php.layer_.emplace_back(lay);
      auto itr = layers.find(lay);
      if (itr == layers.end()) {
        double rin(0), rout(0);
        double zz = HGCalTBParameters::k_ScaleFromDDD * fv.translation().Z();
        if ((sol.shape() == DDSolidShape::ddpolyhedra_rz) || (sol.shape() == DDSolidShape::ddpolyhedra_rrz)) {
          const DDPolyhedra& polyhedra = static_cast<DDPolyhedra>(sol);
          const std::vector<double>& rmin = polyhedra.rMinVec();
          const std::vector<double>& rmax = polyhedra.rMaxVec();
          rin = 0.5 * HGCalTBParameters::k_ScaleFromDDD * (rmin[0] + rmin[1]);
          rout = 0.5 * HGCalTBParameters::k_ScaleFromDDD * (rmax[0] + rmax[1]);
        } else if (sol.shape() == DDSolidShape::ddtubs) {
          const DDTubs& tube = static_cast<DDTubs>(sol);
          rin = HGCalTBParameters::k_ScaleFromDDD * tube.rIn();
          rout = HGCalTBParameters::k_ScaleFromDDD * tube.rOut();
        }
        HGCalTBGeomParameters::layerParameters laypar(rin, rout, zz);
        layers[lay] = laypar;
      }
      DD3Vector x, y, z;
      fv.rotation().GetComponents(x, y, z);
      const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
      const CLHEP::HepRotation hr(rotation);
      double xx = HGCalTBParameters::k_ScaleFromDDD * fv.translation().X();
      if (std::abs(xx) < tolerance)
        xx = 0;
      double yy = HGCalTBParameters::k_ScaleFromDDD * fv.translation().Y();
      if (std::abs(yy) < tolerance)
        yy = 0;
      double zz = HGCalTBParameters::k_ScaleFromDDD * fv.translation().Z();
      const CLHEP::Hep3Vector h3v(xx, yy, zz);
      HGCalTBParameters::hgtrform mytrf;
      mytrf.zp = zp;
      mytrf.lay = lay;
      mytrf.sec = 0;
      mytrf.subsec = 0;
      mytrf.h3v = h3v;
      mytrf.hr = hr;
      trforms.emplace_back(mytrf);
      trformUse.emplace_back(false);
    }
    dodet = fv.next();
  }

  // Then wafers
  // This assumes layers are build starting from 1 (which on 25 Jan 2016, they
  // were) to ensure that new copy numbers are always added to the end of the
  // list.
  std::unordered_map<int32_t, int32_t> copies;
  HGCalTBParameters::layer_map copiesInLayers(layers.size() + 1);
  std::vector<int32_t> wafer2copy;
  std::vector<HGCalTBGeomParameters::cellParameters> wafers;
  std::string attribute = "Volume";
  DDValue val1(attribute, sdTag2, 0.0);
  DDSpecificsMatchesValueFilter filter1{val1};
  DDFilteredView fv1(*cpv, filter1);
  bool ok = fv1.firstChild();
  if (!ok) {
    throw cms::Exception("DDException") << "Attribute " << val1 << " not found but needed.";
  } else {
    dodet = true;
    std::unordered_set<std::string> names;
    while (dodet) {
      const DDSolid& sol = fv1.logicalPart().solid();
      const std::string& name = fv1.logicalPart().name().name();
      std::vector<int> copy = fv1.copyNumbers();
      int nsiz = static_cast<int>(copy.size());
      int wafer = (nsiz > 0) ? copy[nsiz - 1] : 0;
      int layer = (nsiz > 1) ? copy[nsiz - 2] : 0;
      if (nsiz < 2) {
        throw cms::Exception("DDException") << "Funny wafer # " << wafer << " in " << nsiz << " components";
      } else if (layer > static_cast<int>(layers.size())) {
        edm::LogWarning("HGCalGeom") << "Funny wafer # " << wafer << " Layer " << layer << ":" << layers.size()
                                     << " among " << nsiz << " components";
      } else {
        auto itr = copies.find(wafer);
        auto cpy = copiesInLayers[layer].find(wafer);
        if (itr != copies.end() && cpy == copiesInLayers[layer].end()) {
          copiesInLayers[layer][wafer] = itr->second;
        }
        if (itr == copies.end()) {
          copies[wafer] = wafer2copy.size();
          copiesInLayers[layer][wafer] = wafer2copy.size();
          double xx = HGCalTBParameters::k_ScaleFromDDD * fv1.translation().X();
          if (std::abs(xx) < tolerance)
            xx = 0;
          double yy = HGCalTBParameters::k_ScaleFromDDD * fv1.translation().Y();
          if (std::abs(yy) < tolerance)
            yy = 0;
          wafer2copy.emplace_back(wafer);
          GlobalPoint p(xx, yy, HGCalTBParameters::k_ScaleFromDDD * fv1.translation().Z());
          HGCalTBGeomParameters::cellParameters cell(false, wafer, p);
          wafers.emplace_back(cell);
          if (names.count(name) == 0) {
            std::vector<double> zv, rv;
            if (mode == HGCalGeometryMode::Polyhedra) {
              const DDPolyhedra& polyhedra = static_cast<DDPolyhedra>(sol);
              zv = polyhedra.zVec();
              rv = polyhedra.rMaxVec();
            } else {
              const DDExtrudedPolygon& polygon = static_cast<DDExtrudedPolygon>(sol);
              zv = polygon.zVec();
              rv = polygon.xVec();
            }
            php.waferR_ = 2.0 * HGCalTBParameters::k_ScaleFromDDDToG4 * rv[0] * tan30deg_;
            php.waferSize_ = HGCalTBParameters::k_ScaleFromDDD * rv[0];
            double dz = 0.5 * HGCalTBParameters::k_ScaleFromDDDToG4 * (zv[1] - zv[0]);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "Mode " << mode << " R " << php.waferSize_ << ":" << php.waferR_ << " z " << dz;
#endif
            HGCalTBParameters::hgtrap mytr;
            mytr.lay = 1;
            mytr.bl = php.waferR_;
            mytr.tl = php.waferR_;
            mytr.h = php.waferR_;
            mytr.dz = dz;
            mytr.alpha = 0.0;
            mytr.cellSize = waferSize_;
            php.fillModule(mytr, false);
            names.insert(name);
          }
        }
      }
      dodet = fv1.next();
    }
  }

  // Finally the cells
  std::map<int, int> wafertype;
  std::map<int, HGCalTBGeomParameters::cellParameters> cellsf, cellsc;
  DDValue val2(attribute, sdTag3, 0.0);
  DDSpecificsMatchesValueFilter filter2{val2};
  DDFilteredView fv2(*cpv, filter2);
  ok = fv2.firstChild();
  if (!ok) {
    throw cms::Exception("DDException") << "Attribute " << val2 << " not found but needed.";
  } else {
    dodet = true;
    while (dodet) {
      const DDSolid& sol = fv2.logicalPart().solid();
      const std::string& name = sol.name().name();
      std::vector<int> copy = fv2.copyNumbers();
      int nsiz = static_cast<int>(copy.size());
      int cellx = (nsiz > 0) ? copy[nsiz - 1] : 0;
      int wafer = (nsiz > 1) ? copy[nsiz - 2] : 0;
      int cell = HGCalTypes::getUnpackedCell6(cellx);
      int type = HGCalTypes::getUnpackedCellType6(cellx);
      if (type != 1 && type != 2) {
        throw cms::Exception("DDException")
            << "Funny cell # " << cell << " type " << type << " in " << nsiz << " components";
      } else {
        auto ktr = wafertype.find(wafer);
        if (ktr == wafertype.end())
          wafertype[wafer] = type;
        bool newc(false);
        std::map<int, HGCalTBGeomParameters::cellParameters>::iterator itr;
        double cellsize = php.cellSize_[0];
        if (type == 1) {
          itr = cellsf.find(cell);
          newc = (itr == cellsf.end());
        } else {
          itr = cellsc.find(cell);
          newc = (itr == cellsc.end());
          cellsize = php.cellSize_[1];
        }
        if (newc) {
          bool half = (name.find("Half") != std::string::npos);
          double xx = HGCalTBParameters::k_ScaleFromDDD * fv2.translation().X();
          double yy = HGCalTBParameters::k_ScaleFromDDD * fv2.translation().Y();
          if (half) {
            math::XYZPointD p1(-2.0 * cellsize / 9.0, 0, 0);
            math::XYZPointD p2 = fv2.rotation()(p1);
            xx += (HGCalTBParameters::k_ScaleFromDDD * (p2.X()));
            yy += (HGCalTBParameters::k_ScaleFromDDD * (p2.Y()));
#ifdef EDM_ML_DEBUG
            if (std::abs(p2.X()) < HGCalTBParameters::tol)
              p2.SetX(0.0);
            if (std::abs(p2.Z()) < HGCalTBParameters::tol)
              p2.SetZ(0.0);
            edm::LogVerbatim("HGCalGeom") << "Wafer " << wafer << " Type " << type << " Cell " << cellx << " local "
                                          << xx << ":" << yy << " new " << p1 << ":" << p2;
#endif
          }
          HGCalTBGeomParameters::cellParameters cp(half, wafer, GlobalPoint(xx, yy, 0));
          if (type == 1) {
            cellsf[cell] = cp;
          } else {
            cellsc[cell] = cp;
          }
        }
      }
      dodet = fv2.next();
    }
  }

  loadGeometryHexagon(
      layers, trforms, trformUse, copies, copiesInLayers, wafer2copy, wafers, wafertype, cellsf, cellsc, php);
}

void HGCalTBGeomParameters::loadGeometryHexagon(const cms::DDCompactView* cpv,
                                                HGCalTBParameters& php,
                                                const std::string& sdTag1,
                                                const std::string& sdTag2,
                                                const std::string& sdTag3,
                                                HGCalGeometryMode::WaferMode mode) {
  const cms::DDFilter filter("Volume", sdTag1);
  cms::DDFilteredView fv((*cpv), filter);
  std::map<int, HGCalTBGeomParameters::layerParameters> layers;
  std::vector<HGCalTBParameters::hgtrform> trforms;
  std::vector<bool> trformUse;
  std::vector<std::pair<int, int> > trused;

  while (fv.firstChild()) {
    const std::vector<double>& pars = fv.parameters();
    // Layers first
    std::vector<int> copy = fv.copyNos();
    int nsiz = static_cast<int>(copy.size());
    int lay = (nsiz > 0) ? copy[0] : 0;
    int zp = (nsiz > 2) ? copy[2] : -1;
    if (zp != 1)
      zp = -1;
    if (lay == 0) {
      throw cms::Exception("DDException") << "Funny layer # " << lay << " zp " << zp << " in " << nsiz << " components";
    } else {
      if (std::find(php.layer_.begin(), php.layer_.end(), lay) == php.layer_.end())
        php.layer_.emplace_back(lay);
      auto itr = layers.find(lay);
      double zz = HGCalTBParameters::k_ScaleFromDD4hep * fv.translation().Z();
      if (itr == layers.end()) {
        double rin(0), rout(0);
        if (dd4hep::isA<dd4hep::Polyhedra>(fv.solid())) {
          rin = 0.5 * HGCalTBParameters::k_ScaleFromDD4hep * (pars[5] + pars[8]);
          rout = 0.5 * HGCalTBParameters::k_ScaleFromDD4hep * (pars[6] + pars[9]);
        } else if (dd4hep::isA<dd4hep::Tube>(fv.solid())) {
          dd4hep::Tube tubeSeg(fv.solid());
          rin = HGCalTBParameters::k_ScaleFromDD4hep * tubeSeg.rMin();
          rout = HGCalTBParameters::k_ScaleFromDD4hep * tubeSeg.rMax();
        }
        HGCalTBGeomParameters::layerParameters laypar(rin, rout, zz);
        layers[lay] = laypar;
      }
      std::pair<int, int> layz(lay, zp);
      if (std::find(trused.begin(), trused.end(), layz) == trused.end()) {
        trused.emplace_back(layz);
        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
        const CLHEP::HepRotation hr(rotation);
        double xx = HGCalTBParameters::k_ScaleFromDD4hep * fv.translation().X();
        if (std::abs(xx) < tolerance)
          xx = 0;
        double yy = HGCalTBParameters::k_ScaleFromDD4hep * fv.translation().Y();
        if (std::abs(yy) < tolerance)
          yy = 0;
        double zz = HGCalTBParameters::k_ScaleFromDD4hep * fv.translation().Z();
        const CLHEP::Hep3Vector h3v(xx, yy, zz);
        HGCalTBParameters::hgtrform mytrf;
        mytrf.zp = zp;
        mytrf.lay = lay;
        mytrf.sec = 0;
        mytrf.subsec = 0;
        mytrf.h3v = h3v;
        mytrf.hr = hr;
        trforms.emplace_back(mytrf);
        trformUse.emplace_back(false);
      }
    }
  }

  // Then wafers
  // This assumes layers are build starting from 1 (which on 25 Jan 2016, they
  // were) to ensure that new copy numbers are always added to the end of the
  // list.
  std::unordered_map<int32_t, int32_t> copies;
  HGCalTBParameters::layer_map copiesInLayers(layers.size() + 1);
  std::vector<int32_t> wafer2copy;
  std::vector<HGCalTBGeomParameters::cellParameters> wafers;
  const cms::DDFilter filter1("Volume", sdTag2);
  cms::DDFilteredView fv1((*cpv), filter1);
  bool ok = fv1.firstChild();
  if (!ok) {
    throw cms::Exception("DDException") << "Attribute " << sdTag2 << " not found but needed.";
  } else {
    bool dodet = true;
    std::unordered_set<std::string> names;
    while (dodet) {
      const std::string name = static_cast<std::string>(fv1.name());
      std::vector<int> copy = fv1.copyNos();
      int nsiz = static_cast<int>(copy.size());
      int wafer = (nsiz > 0) ? copy[0] : 0;
      int layer = (nsiz > 1) ? copy[1] : 0;
      if (nsiz < 2) {
        throw cms::Exception("DDException") << "Funny wafer # " << wafer << " in " << nsiz << " components";
      } else if (layer > static_cast<int>(layers.size())) {
        edm::LogWarning("HGCalGeom") << "Funny wafer # " << wafer << " Layer " << layer << ":" << layers.size()
                                     << " among " << nsiz << " components";
      } else {
        auto itr = copies.find(wafer);
        auto cpy = copiesInLayers[layer].find(wafer);
        if (itr != copies.end() && cpy == copiesInLayers[layer].end()) {
          copiesInLayers[layer][wafer] = itr->second;
        }
        if (itr == copies.end()) {
          copies[wafer] = wafer2copy.size();
          copiesInLayers[layer][wafer] = wafer2copy.size();
          double xx = HGCalTBParameters::k_ScaleFromDD4hep * fv1.translation().X();
          if (std::abs(xx) < tolerance)
            xx = 0;
          double yy = HGCalTBParameters::k_ScaleFromDD4hep * fv1.translation().Y();
          if (std::abs(yy) < tolerance)
            yy = 0;
          wafer2copy.emplace_back(wafer);
          GlobalPoint p(xx, yy, HGCalTBParameters::k_ScaleFromDD4hep * fv1.translation().Z());
          HGCalTBGeomParameters::cellParameters cell(false, wafer, p);
          wafers.emplace_back(cell);
          if (names.count(name) == 0) {
            double zv[2], rv;
            const std::vector<double>& pars = fv1.parameters();
            if (mode == HGCalGeometryMode::Polyhedra) {
              zv[0] = pars[4];
              zv[1] = pars[7];
              rv = pars[6];
            } else {
              zv[0] = pars[3];
              zv[1] = pars[9];
              rv = pars[4];
            }
            php.waferR_ = 2.0 * HGCalTBParameters::k_ScaleFromDD4hepToG4 * rv * tan30deg_;
            php.waferSize_ = HGCalTBParameters::k_ScaleFromDD4hep * rv;
            double dz = 0.5 * HGCalTBParameters::k_ScaleFromDD4hepToG4 * (zv[1] - zv[0]);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "Mode " << mode << " R " << php.waferSize_ << ":" << php.waferR_ << " z " << dz;
#endif
            HGCalTBParameters::hgtrap mytr;
            mytr.lay = 1;
            mytr.bl = php.waferR_;
            mytr.tl = php.waferR_;
            mytr.h = php.waferR_;
            mytr.dz = dz;
            mytr.alpha = 0.0;
            mytr.cellSize = waferSize_;
            php.fillModule(mytr, false);
            names.insert(name);
          }
        }
      }
      dodet = fv1.firstChild();
    }
  }

  // Finally the cells
  std::map<int, int> wafertype;
  std::map<int, HGCalTBGeomParameters::cellParameters> cellsf, cellsc;
  const cms::DDFilter filter2("Volume", sdTag3);
  cms::DDFilteredView fv2((*cpv), filter2);
  ok = fv2.firstChild();
  if (!ok) {
    throw cms::Exception("DDException") << "Attribute " << sdTag3 << " not found but needed.";
  } else {
    bool dodet = true;
    while (dodet) {
      const std::string name = static_cast<std::string>(fv2.name());
      std::vector<int> copy = fv2.copyNos();
      int nsiz = static_cast<int>(copy.size());
      int cellx = (nsiz > 0) ? copy[0] : 0;
      int wafer = (nsiz > 1) ? copy[1] : 0;
      int cell = HGCalTypes::getUnpackedCell6(cellx);
      int type = HGCalTypes::getUnpackedCellType6(cellx);
      if (type != 1 && type != 2) {
        throw cms::Exception("DDException")
            << "Funny cell # " << cell << " type " << type << " in " << nsiz << " components";
      } else {
        auto ktr = wafertype.find(wafer);
        if (ktr == wafertype.end())
          wafertype[wafer] = type;
        bool newc(false);
        std::map<int, HGCalTBGeomParameters::cellParameters>::iterator itr;
        double cellsize = php.cellSize_[0];
        if (type == 1) {
          itr = cellsf.find(cell);
          newc = (itr == cellsf.end());
        } else {
          itr = cellsc.find(cell);
          newc = (itr == cellsc.end());
          cellsize = php.cellSize_[1];
        }
        if (newc) {
          bool half = (name.find("Half") != std::string::npos);
          double xx = HGCalTBParameters::k_ScaleFromDD4hep * fv2.translation().X();
          double yy = HGCalTBParameters::k_ScaleFromDD4hep * fv2.translation().Y();
          if (half) {
            math::XYZPointD p1(-2.0 * cellsize / 9.0, 0, 0);
            math::XYZPointD p2 = fv2.rotation()(p1);
            xx += (HGCalTBParameters::k_ScaleFromDDD * (p2.X()));
            yy += (HGCalTBParameters::k_ScaleFromDDD * (p2.Y()));
#ifdef EDM_ML_DEBUG
            if (std::abs(p2.X()) < HGCalTBParameters::tol)
              p2.SetX(0.0);
            if (std::abs(p2.Z()) < HGCalTBParameters::tol)
              p2.SetZ(0.0);
            edm::LogVerbatim("HGCalGeom") << "Wafer " << wafer << " Type " << type << " Cell " << cellx << " local "
                                          << xx << ":" << yy << " new " << p1 << ":" << p2;
#endif
          }
          HGCalTBGeomParameters::cellParameters cp(half, wafer, GlobalPoint(xx, yy, 0));
          if (type == 1) {
            cellsf[cell] = cp;
          } else {
            cellsc[cell] = cp;
          }
        }
      }
      dodet = fv2.firstChild();
    }
  }

  loadGeometryHexagon(
      layers, trforms, trformUse, copies, copiesInLayers, wafer2copy, wafers, wafertype, cellsf, cellsc, php);
}

void HGCalTBGeomParameters::loadGeometryHexagon(const std::map<int, HGCalTBGeomParameters::layerParameters>& layers,
                                                std::vector<HGCalTBParameters::hgtrform>& trforms,
                                                std::vector<bool>& trformUse,
                                                const std::unordered_map<int32_t, int32_t>& copies,
                                                const HGCalTBParameters::layer_map& copiesInLayers,
                                                const std::vector<int32_t>& wafer2copy,
                                                const std::vector<HGCalTBGeomParameters::cellParameters>& wafers,
                                                const std::map<int, int>& wafertype,
                                                const std::map<int, HGCalTBGeomParameters::cellParameters>& cellsf,
                                                const std::map<int, HGCalTBGeomParameters::cellParameters>& cellsc,
                                                HGCalTBParameters& php) {
  if (((cellsf.size() + cellsc.size()) == 0) || (wafers.empty()) || (layers.empty())) {
    throw cms::Exception("DDException") << "HGCalTBGeomParameters: mismatch between geometry and specpar: cells "
                                        << cellsf.size() << ":" << cellsc.size() << " wafers " << wafers.size()
                                        << " layers " << layers.size();
  }

  for (unsigned int i = 0; i < layers.size(); ++i) {
    for (auto& layer : layers) {
      if (layer.first == static_cast<int>(i + php.firstLayer_)) {
        php.layerIndex_.emplace_back(i);
        php.rMinLayHex_.emplace_back(layer.second.rmin);
        php.rMaxLayHex_.emplace_back(layer.second.rmax);
        php.zLayerHex_.emplace_back(layer.second.zpos);
        break;
      }
    }
  }

  for (unsigned int i = 0; i < php.layer_.size(); ++i) {
    for (unsigned int i1 = 0; i1 < trforms.size(); ++i1) {
      if (!trformUse[i1] && php.layerGroup_[trforms[i1].lay - 1] == static_cast<int>(i + 1)) {
        trforms[i1].h3v *= static_cast<double>(HGCalTBParameters::k_ScaleFromDDD);
        trforms[i1].lay = (i + 1);
        trformUse[i1] = true;
        php.fillTrForm(trforms[i1]);
        int nz(1);
        for (unsigned int i2 = i1 + 1; i2 < trforms.size(); ++i2) {
          if (!trformUse[i2] && trforms[i2].zp == trforms[i1].zp &&
              php.layerGroup_[trforms[i2].lay - 1] == static_cast<int>(i + 1)) {
            php.addTrForm(trforms[i2].h3v);
            nz++;
            trformUse[i2] = true;
          }
        }
        if (nz > 0) {
          php.scaleTrForm(double(1.0 / nz));
        }
      }
    }
  }

  double rmin = HGCalTBParameters::k_ScaleFromDDD * php.waferR_;
  for (unsigned i = 0; i < wafer2copy.size(); ++i) {
    php.waferCopy_.emplace_back(wafer2copy[i]);
    php.waferPosX_.emplace_back(wafers[i].xyz.x());
    php.waferPosY_.emplace_back(wafers[i].xyz.y());
    auto ktr = wafertype.find(wafer2copy[i]);
    int typet = (ktr == wafertype.end()) ? 0 : (ktr->second);
    php.waferTypeT_.emplace_back(typet);
    double r = wafers[i].xyz.perp();
    int type(3);
    for (int k = 1; k < 4; ++k) {
      if ((r + rmin) <= php.boundR_[k]) {
        type = k;
        break;
      }
    }
    php.waferTypeL_.emplace_back(type);
  }
  php.copiesInLayers_ = copiesInLayers;
  php.nSectors_ = static_cast<int>(php.waferCopy_.size());

  std::vector<HGCalTBGeomParameters::cellParameters>::const_iterator itrf = wafers.end();
  for (unsigned int i = 0; i < cellsf.size(); ++i) {
    auto itr = cellsf.find(i);
    if (itr == cellsf.end()) {
      throw cms::Exception("DDException") << "HGCalTBGeomParameters: missing info for fine cell number " << i;
    } else {
      double xx = (itr->second).xyz.x();
      double yy = (itr->second).xyz.y();
      int waf = (itr->second).wafer;
      std::pair<double, double> xy = cellPosition(wafers, itrf, waf, xx, yy);
      php.cellFineX_.emplace_back(xy.first);
      php.cellFineY_.emplace_back(xy.second);
      php.cellFineHalf_.emplace_back((itr->second).half);
    }
  }
  itrf = wafers.end();
  for (unsigned int i = 0; i < cellsc.size(); ++i) {
    auto itr = cellsc.find(i);
    if (itr == cellsc.end()) {
      throw cms::Exception("DDException") << "HGCalTBGeomParameters: missing info for coarse cell number " << i;
    } else {
      double xx = (itr->second).xyz.x();
      double yy = (itr->second).xyz.y();
      int waf = (itr->second).wafer;
      std::pair<double, double> xy = cellPosition(wafers, itrf, waf, xx, yy);
      php.cellCoarseX_.emplace_back(xy.first);
      php.cellCoarseY_.emplace_back(xy.second);
      php.cellCoarseHalf_.emplace_back((itr->second).half);
    }
  }
  int depth(0);
  for (unsigned int i = 0; i < php.layerGroup_.size(); ++i) {
    bool first(true);
    for (unsigned int k = 0; k < php.layerGroup_.size(); ++k) {
      if (php.layerGroup_[k] == static_cast<int>(i + 1)) {
        if (first) {
          php.depth_.emplace_back(i + 1);
          php.depthIndex_.emplace_back(depth);
          php.depthLayerF_.emplace_back(k);
          ++depth;
          first = false;
        }
      }
    }
  }
  HGCalTBParameters::hgtrap mytr = php.getModule(0, false);
  mytr.bl *= HGCalTBParameters::k_ScaleFromDDD;
  mytr.tl *= HGCalTBParameters::k_ScaleFromDDD;
  mytr.h *= HGCalTBParameters::k_ScaleFromDDD;
  mytr.dz *= HGCalTBParameters::k_ScaleFromDDD;
  mytr.cellSize *= HGCalTBParameters::k_ScaleFromDDD;
  double dz = mytr.dz;
  php.fillModule(mytr, true);
  mytr.dz = 2 * dz;
  php.fillModule(mytr, true);
  mytr.dz = 3 * dz;
  php.fillModule(mytr, true);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters finds " << php.zLayerHex_.size() << " layers";
  for (unsigned int i = 0; i < php.zLayerHex_.size(); ++i) {
    int k = php.layerIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Layer[" << i << ":" << k << ":" << php.layer_[k]
                                  << "] with r = " << php.rMinLayHex_[i] << ":" << php.rMaxLayHex_[i]
                                  << " at z = " << php.zLayerHex_[i];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters has " << php.depthIndex_.size() << " depths";
  for (unsigned int i = 0; i < php.depthIndex_.size(); ++i) {
    int k = php.depthIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Reco Layer[" << i << ":" << k << "]  First Layer " << php.depthLayerF_[i]
                                  << " Depth " << php.depth_[k];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters finds " << php.nSectors_ << " wafers";
  for (unsigned int i = 0; i < php.waferCopy_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << ": " << php.waferCopy_[i] << "] type " << php.waferTypeL_[i]
                                  << ":" << php.waferTypeT_[i] << " at (" << php.waferPosX_[i] << ","
                                  << php.waferPosY_[i] << ",0)";
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: wafer radius " << php.waferR_ << " and dimensions of the "
                                << "wafers:";
  edm::LogVerbatim("HGCalGeom") << "Sim[0] " << php.moduleLayS_[0] << " dx " << php.moduleBlS_[0] << ":"
                                << php.moduleTlS_[0] << " dy " << php.moduleHS_[0] << " dz " << php.moduleDzS_[0]
                                << " alpha " << php.moduleAlphaS_[0];
  for (unsigned int k = 0; k < php.moduleLayR_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Rec[" << k << "] " << php.moduleLayR_[k] << " dx " << php.moduleBlR_[k] << ":"
                                  << php.moduleTlR_[k] << " dy " << php.moduleHR_[k] << " dz " << php.moduleDzR_[k]
                                  << " alpha " << php.moduleAlphaR_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters finds " << php.cellFineX_.size() << " fine cells in a  wafer";
  for (unsigned int i = 0; i < php.cellFineX_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Fine Cell[" << i << "] at (" << php.cellFineX_[i] << "," << php.cellFineY_[i]
                                  << ",0)";
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters finds " << php.cellCoarseX_.size()
                                << " coarse cells in a wafer";
  for (unsigned int i = 0; i < php.cellCoarseX_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Coarse Cell[" << i << "] at (" << php.cellCoarseX_[i] << ","
                                  << php.cellCoarseY_[i] << ",0)";
  edm::LogVerbatim("HGCalGeom") << "Obtained " << php.trformIndex_.size() << " transformation matrices";
  for (unsigned int k = 0; k < php.trformIndex_.size(); ++k) {
    edm::LogVerbatim("HGCalGeom") << "Matrix[" << k << "] (" << std::hex << php.trformIndex_[k] << std::dec
                                  << ") Translation (" << php.trformTranX_[k] << ", " << php.trformTranY_[k] << ", "
                                  << php.trformTranZ_[k] << " Rotation (" << php.trformRotXX_[k] << ", "
                                  << php.trformRotYX_[k] << ", " << php.trformRotZX_[k] << ", " << php.trformRotXY_[k]
                                  << ", " << php.trformRotYY_[k] << ", " << php.trformRotZY_[k] << ", "
                                  << php.trformRotXZ_[k] << ", " << php.trformRotYZ_[k] << ", " << php.trformRotZZ_[k]
                                  << ")";
  }
  edm::LogVerbatim("HGCalGeom") << "Dump copiesInLayers for " << php.copiesInLayers_.size() << " layers";
  for (unsigned int k = 0; k < php.copiesInLayers_.size(); ++k) {
    const auto& theModules = php.copiesInLayers_[k];
    edm::LogVerbatim("HGCalGeom") << "Layer " << k << ":" << theModules.size();
    int k2(0);
    for (std::unordered_map<int, int>::const_iterator itr = theModules.begin(); itr != theModules.end(); ++itr, ++k2) {
      edm::LogVerbatim("HGCalGeom") << "[" << k2 << "] " << itr->first << ":" << itr->second;
    }
  }
#endif
}

void HGCalTBGeomParameters::loadSpecParsHexagon(const DDFilteredView& fv,
                                                HGCalTBParameters& php,
                                                const DDCompactView* cpv,
                                                const std::string& sdTag1,
                                                const std::string& sdTag2) {
  DDsvalues_type sv(fv.mergedSpecifics());
  php.boundR_ = getDDDArray("RadiusBound", sv, 4);
  rescale(php.boundR_, HGCalTBParameters::k_ScaleFromDDD);
  php.rLimit_ = getDDDArray("RadiusLimits", sv, 2);
  rescale(php.rLimit_, HGCalTBParameters::k_ScaleFromDDD);
  php.levelT_ = dbl_to_int(getDDDArray("LevelTop", sv, 0));

  // Grouping of layers
  php.layerGroup_ = dbl_to_int(getDDDArray("GroupingZFine", sv, 0));
  php.layerGroupM_ = dbl_to_int(getDDDArray("GroupingZMid", sv, 0));
  php.layerGroupO_ = dbl_to_int(getDDDArray("GroupingZOut", sv, 0));
  php.slopeMin_ = getDDDArray("Slope", sv, 1);

  // Wafer size
  std::string attribute = "Volume";
  DDSpecificsMatchesValueFilter filter1{DDValue(attribute, sdTag1, 0.0)};
  DDFilteredView fv1(*cpv, filter1);
  if (fv1.firstChild()) {
    DDsvalues_type sv(fv1.mergedSpecifics());
    const auto& dummy = getDDDArray("WaferSize", sv, 0);
    waferSize_ = dummy[0];
  }

  // Cell size
  DDSpecificsMatchesValueFilter filter2{DDValue(attribute, sdTag2, 0.0)};
  DDFilteredView fv2(*cpv, filter2);
  if (fv2.firstChild()) {
    DDsvalues_type sv(fv2.mergedSpecifics());
    php.cellSize_ = getDDDArray("CellSize", sv, 0);
  }

  loadSpecParsHexagon(php);
}

void HGCalTBGeomParameters::loadSpecParsHexagon(const cms::DDFilteredView& fv,
                                                HGCalTBParameters& php,
                                                const std::string& sdTag1,
                                                const std::string& sdTag2,
                                                const std::string& sdTag3,
                                                const std::string& sdTag4) {
  php.boundR_ = fv.get<std::vector<double> >(sdTag4, "RadiusBound");
  rescale(php.boundR_, HGCalTBParameters::k_ScaleFromDD4hep);
  php.rLimit_ = fv.get<std::vector<double> >(sdTag4, "RadiusLimits");
  rescale(php.rLimit_, HGCalTBParameters::k_ScaleFromDD4hep);
  php.levelT_ = dbl_to_int(fv.get<std::vector<double> >(sdTag4, "LevelTop"));

  // Grouping of layers
  php.layerGroup_ = dbl_to_int(fv.get<std::vector<double> >(sdTag1, "GroupingZFine"));
  php.layerGroupM_ = dbl_to_int(fv.get<std::vector<double> >(sdTag1, "GroupingZMid"));
  php.layerGroupO_ = dbl_to_int(fv.get<std::vector<double> >(sdTag1, "GroupingZOut"));
  php.slopeMin_ = fv.get<std::vector<double> >(sdTag4, "Slope");
  if (php.slopeMin_.empty())
    php.slopeMin_.emplace_back(0);

  // Wafer size
  const auto& dummy = fv.get<std::vector<double> >(sdTag2, "WaferSize");
  waferSize_ = dummy[0] * HGCalTBParameters::k_ScaleFromDD4hepToG4;

  // Cell size
  php.cellSize_ = fv.get<std::vector<double> >(sdTag3, "CellSize");
  rescale(php.cellSize_, HGCalTBParameters::k_ScaleFromDD4hepToG4);

  loadSpecParsHexagon(php);
}

void HGCalTBGeomParameters::loadSpecParsHexagon(const HGCalTBParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: wafer radius ranges"
                                << " for cell grouping " << php.boundR_[0] << ":" << php.boundR_[1] << ":"
                                << php.boundR_[2] << ":" << php.boundR_[3];
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: Minimum/maximum R " << php.rLimit_[0] << ":"
                                << php.rLimit_[1];
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: LevelTop " << php.levelT_[0];
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: minimum slope " << php.slopeMin_[0]
                                << " and layer groupings "
                                << "for the 3 ranges:";
  for (unsigned int k = 0; k < php.layerGroup_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.layerGroup_[k] << ":" << php.layerGroupM_[k] << ":"
                                  << php.layerGroupO_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: Wafer Size: " << waferSize_;
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: " << php.cellSize_.size() << " cells of sizes:";
  for (unsigned int k = 0; k < php.cellSize_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << php.cellSize_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeomParameters: First Layer " << php.firstLayer_;
#endif
}

void HGCalTBGeomParameters::loadWaferHexagon(HGCalTBParameters& php) {
  double waferW(HGCalTBParameters::k_ScaleFromDDD * waferSize_), rmin(HGCalTBParameters::k_ScaleFromDDD * php.waferR_);
  double rin(php.rLimit_[0]), rout(php.rLimit_[1]), rMaxFine(php.boundR_[1]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Input waferWidth " << waferW << ":" << rmin << " R Limits: " << rin << ":" << rout
                                << " Fine " << rMaxFine;
#endif
  // Clear the vectors
  php.waferCopy_.clear();
  php.waferTypeL_.clear();
  php.waferTypeT_.clear();
  php.waferPosX_.clear();
  php.waferPosY_.clear();
  double dx = 0.5 * waferW;
  double dy = 3.0 * dx * tan(30._deg);
  double rr = 2.0 * dx * tan(30._deg);
  int ncol = static_cast<int>(2.0 * rout / waferW) + 1;
  int nrow = static_cast<int>(rout / (waferW * tan(30._deg))) + 1;
  int ns2 = (2 * ncol + 1) * (2 * nrow + 1) * php.layer_.size();
  int incm(0), inrm(0), kount(0), ntot(0);
  HGCalTBParameters::layer_map copiesInLayers(php.layer_.size() + 1);
  HGCalTBParameters::waferT_map waferTypes(ns2 + 1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Row " << nrow << " Column " << ncol;
#endif
  for (int nr = -nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc = -ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr % 2 == inc % 2) {
        double xpos = nc * dx;
        double ypos = nr * dy;
        std::pair<int, int> corner = HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, rin, rout, true);
        double rpos = std::sqrt(xpos * xpos + ypos * ypos);
        int typet = (rpos < rMaxFine) ? 1 : 2;
        int typel(3);
        for (int k = 1; k < 4; ++k) {
          if ((rpos + rmin) <= php.boundR_[k]) {
            typel = k;
            break;
          }
        }
        ++ntot;
        if (corner.first > 0) {
          int copy = HGCalTypes::packTypeUV(typel, nc, nr);
          if (inc > incm)
            incm = inc;
          if (inr > inrm)
            inrm = inr;
          kount++;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << kount << ":" << ntot << " Copy " << copy << " Type " << typel << ":" << typet
                                        << " Location " << corner.first << " Position " << xpos << ":" << ypos
                                        << " Layers " << php.layer_.size();
#endif
          php.waferCopy_.emplace_back(copy);
          php.waferTypeL_.emplace_back(typel);
          php.waferTypeT_.emplace_back(typet);
          php.waferPosX_.emplace_back(xpos);
          php.waferPosY_.emplace_back(ypos);
          for (unsigned int il = 0; il < php.layer_.size(); ++il) {
            std::pair<int, int> corner =
                HGCalGeomTools::waferCorner(xpos, ypos, dx, rr, php.rMinLayHex_[il], php.rMaxLayHex_[il], true);
            if (corner.first > 0) {
              auto cpy = copiesInLayers[php.layer_[il]].find(copy);
              if (cpy == copiesInLayers[php.layer_[il]].end())
                copiesInLayers[php.layer_[il]][copy] =
                    ((corner.first == static_cast<int>(HGCalTBParameters::k_CornerSize)) ? php.waferCopy_.size() : -1);
            }
            if ((corner.first > 0) && (corner.first < static_cast<int>(HGCalTBParameters::k_CornerSize))) {
              int wl = HGCalWaferIndex::waferIndex(php.layer_[il], copy, 0, true);
              waferTypes[wl] = corner;
            }
          }
        }
      }
    }
  }
  php.copiesInLayers_ = copiesInLayers;
  php.waferTypes_ = waferTypes;
  php.nSectors_ = static_cast<int>(php.waferCopy_.size());
  php.waferUVMax_ = 0;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalWaferHexagon: # of columns " << incm << " # of rows " << inrm << " and "
                                << kount << ":" << ntot << " wafers; R " << rin << ":" << rout;
  edm::LogVerbatim("HGCalGeom") << "Dump copiesInLayers for " << php.copiesInLayers_.size() << " layers";
  for (unsigned int k = 0; k < copiesInLayers.size(); ++k) {
    const auto& theModules = copiesInLayers[k];
    edm::LogVerbatim("HGCalGeom") << "Layer " << k << ":" << theModules.size();
    int k2(0);
    for (std::unordered_map<int, int>::const_iterator itr = theModules.begin(); itr != theModules.end(); ++itr, ++k2) {
      edm::LogVerbatim("HGCalGeom") << "[" << k2 << "] " << itr->first << ":" << itr->second;
    }
  }
#endif
}

void HGCalTBGeomParameters::loadCellParsHexagon(const DDCompactView* cpv, HGCalTBParameters& php) {
  // Special parameters for cell parameters
  std::string attribute = "OnlyForHGCalNumbering";
  DDSpecificsHasNamedValueFilter filter1{attribute};
  DDFilteredView fv1(*cpv, filter1);
  bool ok = fv1.firstChild();

  if (ok) {
    php.cellFine_ = dbl_to_int(cpv->vector("waferFine"));
    php.cellCoarse_ = dbl_to_int(cpv->vector("waferCoarse"));
  }

  loadCellParsHexagon(php);
}

void HGCalTBGeomParameters::loadCellParsHexagon(const cms::DDVectorsMap& vmap, HGCalTBParameters& php) {
  for (auto const& it : vmap) {
    if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "waferFine")) {
      for (const auto& i : it.second)
        php.cellFine_.emplace_back(std::round(i));
    } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "waferCoarse")) {
      for (const auto& i : it.second)
        php.cellCoarse_.emplace_back(std::round(i));
    }
  }

  loadCellParsHexagon(php);
}

void HGCalTBGeomParameters::loadCellParsHexagon(const HGCalTBParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalLoadCellPars: " << php.cellFine_.size() << " rows for fine cells";
  for (unsigned int k = 0; k < php.cellFine_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "]: " << php.cellFine_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalLoadCellPars: " << php.cellCoarse_.size() << " rows for coarse cells";
  for (unsigned int k = 0; k < php.cellCoarse_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "]: " << php.cellCoarse_[k];
#endif
}

std::vector<double> HGCalTBGeomParameters::getDDDArray(const std::string& str,
                                                       const DDsvalues_type& sv,
                                                       const int nmin) {
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        throw cms::Exception("DDException")
            << "HGCalTBGeomParameters:  # of " << str << " bins " << nval << " < " << nmin << " ==> illegal";
      }
    } else {
      if (nval < 1 && nmin == 0) {
        throw cms::Exception("DDException")
            << "HGCalTBGeomParameters: # of " << str << " bins " << nval << " < 1 ==> illegal"
            << " (nmin=" << nmin << ")";
      }
    }
    return fvec;
  } else {
    if (nmin >= 0) {
      throw cms::Exception("DDException") << "HGCalTBGeomParameters: cannot get array " << str;
    }
    std::vector<double> fvec;
    return fvec;
  }
}

std::pair<double, double> HGCalTBGeomParameters::cellPosition(
    const std::vector<HGCalTBGeomParameters::cellParameters>& wafers,
    std::vector<HGCalTBGeomParameters::cellParameters>::const_iterator& itrf,
    int wafer,
    double xx,
    double yy) {
  if (itrf == wafers.end()) {
    for (std::vector<HGCalTBGeomParameters::cellParameters>::const_iterator itr = wafers.begin(); itr != wafers.end();
         ++itr) {
      if (itr->wafer == wafer) {
        itrf = itr;
        break;
      }
    }
  }
  double dx(0), dy(0);
  if (itrf != wafers.end()) {
    dx = (xx - itrf->xyz.x());
    if (std::abs(dx) < tolerance)
      dx = 0;
    dy = (yy - itrf->xyz.y());
    if (std::abs(dy) < tolerance)
      dy = 0;
  }
  return std::make_pair(dx, dy);
}

void HGCalTBGeomParameters::rescale(std::vector<double>& v, const double s) {
  std::for_each(v.begin(), v.end(), [s](double& n) { n *= s; });
}

void HGCalTBGeomParameters::resetZero(std::vector<double>& v) {
  for (auto& n : v) {
    if (std::abs(n) < tolmin)
      n = 0;
  }
}
