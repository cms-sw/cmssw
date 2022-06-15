#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"

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
#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

const double tolerance = 0.001;
const double tolmin = 1.e-20;

HGCalGeomParameters::HGCalGeomParameters() : sqrt3_(std::sqrt(3.0)) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters::HGCalGeomParameters "
                                << "constructor";
#endif
}

HGCalGeomParameters::~HGCalGeomParameters() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters::destructed!!!";
#endif
}

void HGCalGeomParameters::loadGeometryHexagon(const DDFilteredView& _fv,
                                              HGCalParameters& php,
                                              const std::string& sdTag1,
                                              const DDCompactView* cpv,
                                              const std::string& sdTag2,
                                              const std::string& sdTag3,
                                              HGCalGeometryMode::WaferMode mode) {
  DDFilteredView fv = _fv;
  bool dodet(true);
  std::map<int, HGCalGeomParameters::layerParameters> layers;
  std::vector<HGCalParameters::hgtrform> trforms;
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
        double zz = HGCalParameters::k_ScaleFromDDD * fv.translation().Z();
        if ((sol.shape() == DDSolidShape::ddpolyhedra_rz) || (sol.shape() == DDSolidShape::ddpolyhedra_rrz)) {
          const DDPolyhedra& polyhedra = static_cast<DDPolyhedra>(sol);
          const std::vector<double>& rmin = polyhedra.rMinVec();
          const std::vector<double>& rmax = polyhedra.rMaxVec();
          rin = 0.5 * HGCalParameters::k_ScaleFromDDD * (rmin[0] + rmin[1]);
          rout = 0.5 * HGCalParameters::k_ScaleFromDDD * (rmax[0] + rmax[1]);
        } else if (sol.shape() == DDSolidShape::ddtubs) {
          const DDTubs& tube = static_cast<DDTubs>(sol);
          rin = HGCalParameters::k_ScaleFromDDD * tube.rIn();
          rout = HGCalParameters::k_ScaleFromDDD * tube.rOut();
        }
        HGCalGeomParameters::layerParameters laypar(rin, rout, zz);
        layers[lay] = laypar;
      }
      DD3Vector x, y, z;
      fv.rotation().GetComponents(x, y, z);
      const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
      const CLHEP::HepRotation hr(rotation);
      double xx = HGCalParameters::k_ScaleFromDDD * fv.translation().X();
      if (std::abs(xx) < tolerance)
        xx = 0;
      double yy = HGCalParameters::k_ScaleFromDDD * fv.translation().Y();
      if (std::abs(yy) < tolerance)
        yy = 0;
      double zz = HGCalParameters::k_ScaleFromDDD * fv.translation().Z();
      const CLHEP::Hep3Vector h3v(xx, yy, zz);
      HGCalParameters::hgtrform mytrf;
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
  HGCalParameters::layer_map copiesInLayers(layers.size() + 1);
  std::vector<int32_t> wafer2copy;
  std::vector<HGCalGeomParameters::cellParameters> wafers;
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
          double xx = HGCalParameters::k_ScaleFromDDD * fv1.translation().X();
          if (std::abs(xx) < tolerance)
            xx = 0;
          double yy = HGCalParameters::k_ScaleFromDDD * fv1.translation().Y();
          if (std::abs(yy) < tolerance)
            yy = 0;
          wafer2copy.emplace_back(wafer);
          GlobalPoint p(xx, yy, HGCalParameters::k_ScaleFromDDD * fv1.translation().Z());
          HGCalGeomParameters::cellParameters cell(false, wafer, p);
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
            php.waferR_ = 2.0 * HGCalParameters::k_ScaleFromDDDToG4 * rv[0] * tan30deg_;
            php.waferSize_ = HGCalParameters::k_ScaleFromDDD * rv[0];
            double dz = 0.5 * HGCalParameters::k_ScaleFromDDDToG4 * (zv[1] - zv[0]);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "Mode " << mode << " R " << php.waferSize_ << ":" << php.waferR_ << " z " << dz;
#endif
            HGCalParameters::hgtrap mytr;
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
  std::map<int, HGCalGeomParameters::cellParameters> cellsf, cellsc;
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
        std::map<int, HGCalGeomParameters::cellParameters>::iterator itr;
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
          double xx = HGCalParameters::k_ScaleFromDDD * fv2.translation().X();
          double yy = HGCalParameters::k_ScaleFromDDD * fv2.translation().Y();
          if (half) {
            math::XYZPointD p1(-2.0 * cellsize / 9.0, 0, 0);
            math::XYZPointD p2 = fv2.rotation()(p1);
            xx += (HGCalParameters::k_ScaleFromDDD * (p2.X()));
            yy += (HGCalParameters::k_ScaleFromDDD * (p2.Y()));
#ifdef EDM_ML_DEBUG
            if (std::abs(p2.X()) < HGCalParameters::tol)
              p2.SetX(0.0);
            if (std::abs(p2.Z()) < HGCalParameters::tol)
              p2.SetZ(0.0);
            edm::LogVerbatim("HGCalGeom") << "Wafer " << wafer << " Type " << type << " Cell " << cellx << " local "
                                          << xx << ":" << yy << " new " << p1 << ":" << p2;
#endif
          }
          HGCalGeomParameters::cellParameters cp(half, wafer, GlobalPoint(xx, yy, 0));
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

void HGCalGeomParameters::loadGeometryHexagon(const cms::DDCompactView* cpv,
                                              HGCalParameters& php,
                                              const std::string& sdTag1,
                                              const std::string& sdTag2,
                                              const std::string& sdTag3,
                                              HGCalGeometryMode::WaferMode mode) {
  const cms::DDFilter filter("Volume", sdTag1);
  cms::DDFilteredView fv((*cpv), filter);
  std::map<int, HGCalGeomParameters::layerParameters> layers;
  std::vector<HGCalParameters::hgtrform> trforms;
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
      double zz = HGCalParameters::k_ScaleFromDD4hep * fv.translation().Z();
      if (itr == layers.end()) {
        double rin(0), rout(0);
        if (dd4hep::isA<dd4hep::Polyhedra>(fv.solid())) {
          rin = 0.5 * HGCalParameters::k_ScaleFromDD4hep * (pars[5] + pars[8]);
          rout = 0.5 * HGCalParameters::k_ScaleFromDD4hep * (pars[6] + pars[9]);
        } else if (dd4hep::isA<dd4hep::Tube>(fv.solid())) {
          dd4hep::Tube tubeSeg(fv.solid());
          rin = HGCalParameters::k_ScaleFromDD4hep * tubeSeg.rMin();
          rout = HGCalParameters::k_ScaleFromDD4hep * tubeSeg.rMax();
        }
        HGCalGeomParameters::layerParameters laypar(rin, rout, zz);
        layers[lay] = laypar;
      }
      std::pair<int, int> layz(lay, zp);
      if (std::find(trused.begin(), trused.end(), layz) == trused.end()) {
        trused.emplace_back(layz);
        DD3Vector x, y, z;
        fv.rotation().GetComponents(x, y, z);
        const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
        const CLHEP::HepRotation hr(rotation);
        double xx = HGCalParameters::k_ScaleFromDD4hep * fv.translation().X();
        if (std::abs(xx) < tolerance)
          xx = 0;
        double yy = HGCalParameters::k_ScaleFromDD4hep * fv.translation().Y();
        if (std::abs(yy) < tolerance)
          yy = 0;
        double zz = HGCalParameters::k_ScaleFromDD4hep * fv.translation().Z();
        const CLHEP::Hep3Vector h3v(xx, yy, zz);
        HGCalParameters::hgtrform mytrf;
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
  HGCalParameters::layer_map copiesInLayers(layers.size() + 1);
  std::vector<int32_t> wafer2copy;
  std::vector<HGCalGeomParameters::cellParameters> wafers;
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
          double xx = HGCalParameters::k_ScaleFromDD4hep * fv1.translation().X();
          if (std::abs(xx) < tolerance)
            xx = 0;
          double yy = HGCalParameters::k_ScaleFromDD4hep * fv1.translation().Y();
          if (std::abs(yy) < tolerance)
            yy = 0;
          wafer2copy.emplace_back(wafer);
          GlobalPoint p(xx, yy, HGCalParameters::k_ScaleFromDD4hep * fv1.translation().Z());
          HGCalGeomParameters::cellParameters cell(false, wafer, p);
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
            php.waferR_ = 2.0 * HGCalParameters::k_ScaleFromDD4hepToG4 * rv * tan30deg_;
            php.waferSize_ = HGCalParameters::k_ScaleFromDD4hep * rv;
            double dz = 0.5 * HGCalParameters::k_ScaleFromDD4hepToG4 * (zv[1] - zv[0]);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "Mode " << mode << " R " << php.waferSize_ << ":" << php.waferR_ << " z " << dz;
#endif
            HGCalParameters::hgtrap mytr;
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
  std::map<int, HGCalGeomParameters::cellParameters> cellsf, cellsc;
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
        std::map<int, HGCalGeomParameters::cellParameters>::iterator itr;
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
          double xx = HGCalParameters::k_ScaleFromDD4hep * fv2.translation().X();
          double yy = HGCalParameters::k_ScaleFromDD4hep * fv2.translation().Y();
          if (half) {
            math::XYZPointD p1(-2.0 * cellsize / 9.0, 0, 0);
            math::XYZPointD p2 = fv2.rotation()(p1);
            xx += (HGCalParameters::k_ScaleFromDDD * (p2.X()));
            yy += (HGCalParameters::k_ScaleFromDDD * (p2.Y()));
#ifdef EDM_ML_DEBUG
            if (std::abs(p2.X()) < HGCalParameters::tol)
              p2.SetX(0.0);
            if (std::abs(p2.Z()) < HGCalParameters::tol)
              p2.SetZ(0.0);
            edm::LogVerbatim("HGCalGeom") << "Wafer " << wafer << " Type " << type << " Cell " << cellx << " local "
                                          << xx << ":" << yy << " new " << p1 << ":" << p2;
#endif
          }
          HGCalGeomParameters::cellParameters cp(half, wafer, GlobalPoint(xx, yy, 0));
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

void HGCalGeomParameters::loadGeometryHexagon(const std::map<int, HGCalGeomParameters::layerParameters>& layers,
                                              std::vector<HGCalParameters::hgtrform>& trforms,
                                              std::vector<bool>& trformUse,
                                              const std::unordered_map<int32_t, int32_t>& copies,
                                              const HGCalParameters::layer_map& copiesInLayers,
                                              const std::vector<int32_t>& wafer2copy,
                                              const std::vector<HGCalGeomParameters::cellParameters>& wafers,
                                              const std::map<int, int>& wafertype,
                                              const std::map<int, HGCalGeomParameters::cellParameters>& cellsf,
                                              const std::map<int, HGCalGeomParameters::cellParameters>& cellsc,
                                              HGCalParameters& php) {
  if (((cellsf.size() + cellsc.size()) == 0) || (wafers.empty()) || (layers.empty())) {
    throw cms::Exception("DDException") << "HGCalGeomParameters: mismatch between geometry and specpar: cells "
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
        trforms[i1].h3v *= static_cast<double>(HGCalParameters::k_ScaleFromDDD);
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

  double rmin = HGCalParameters::k_ScaleFromDDD * php.waferR_;
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

  std::vector<HGCalGeomParameters::cellParameters>::const_iterator itrf = wafers.end();
  for (unsigned int i = 0; i < cellsf.size(); ++i) {
    auto itr = cellsf.find(i);
    if (itr == cellsf.end()) {
      throw cms::Exception("DDException") << "HGCalGeomParameters: missing info for fine cell number " << i;
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
      throw cms::Exception("DDException") << "HGCalGeomParameters: missing info for coarse cell number " << i;
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
  HGCalParameters::hgtrap mytr = php.getModule(0, false);
  mytr.bl *= HGCalParameters::k_ScaleFromDDD;
  mytr.tl *= HGCalParameters::k_ScaleFromDDD;
  mytr.h *= HGCalParameters::k_ScaleFromDDD;
  mytr.dz *= HGCalParameters::k_ScaleFromDDD;
  mytr.cellSize *= HGCalParameters::k_ScaleFromDDD;
  double dz = mytr.dz;
  php.fillModule(mytr, true);
  mytr.dz = 2 * dz;
  php.fillModule(mytr, true);
  mytr.dz = 3 * dz;
  php.fillModule(mytr, true);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " << php.zLayerHex_.size() << " layers";
  for (unsigned int i = 0; i < php.zLayerHex_.size(); ++i) {
    int k = php.layerIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Layer[" << i << ":" << k << ":" << php.layer_[k]
                                  << "] with r = " << php.rMinLayHex_[i] << ":" << php.rMaxLayHex_[i]
                                  << " at z = " << php.zLayerHex_[i];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters has " << php.depthIndex_.size() << " depths";
  for (unsigned int i = 0; i < php.depthIndex_.size(); ++i) {
    int k = php.depthIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Reco Layer[" << i << ":" << k << "]  First Layer " << php.depthLayerF_[i]
                                  << " Depth " << php.depth_[k];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " << php.nSectors_ << " wafers";
  for (unsigned int i = 0; i < php.waferCopy_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << i << ": " << php.waferCopy_[i] << "] type " << php.waferTypeL_[i]
                                  << ":" << php.waferTypeT_[i] << " at (" << php.waferPosX_[i] << ","
                                  << php.waferPosY_[i] << ",0)";
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: wafer radius " << php.waferR_ << " and dimensions of the "
                                << "wafers:";
  edm::LogVerbatim("HGCalGeom") << "Sim[0] " << php.moduleLayS_[0] << " dx " << php.moduleBlS_[0] << ":"
                                << php.moduleTlS_[0] << " dy " << php.moduleHS_[0] << " dz " << php.moduleDzS_[0]
                                << " alpha " << php.moduleAlphaS_[0];
  for (unsigned int k = 0; k < php.moduleLayR_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Rec[" << k << "] " << php.moduleLayR_[k] << " dx " << php.moduleBlR_[k] << ":"
                                  << php.moduleTlR_[k] << " dy " << php.moduleHR_[k] << " dz " << php.moduleDzR_[k]
                                  << " alpha " << php.moduleAlphaR_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " << php.cellFineX_.size() << " fine cells in a  wafer";
  for (unsigned int i = 0; i < php.cellFineX_.size(); ++i)
    edm::LogVerbatim("HGCalGeom") << "Fine Cell[" << i << "] at (" << php.cellFineX_[i] << "," << php.cellFineY_[i]
                                  << ",0)";
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " << php.cellCoarseX_.size()
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

void HGCalGeomParameters::loadGeometryHexagon8(const DDFilteredView& _fv, HGCalParameters& php, int firstLayer) {
  DDFilteredView fv = _fv;
  bool dodet(true);
  std::map<int, HGCalGeomParameters::layerParameters> layers;
  std::map<std::pair<int, int>, HGCalParameters::hgtrform> trforms;
  int levelTop = 3 + std::max(php.levelT_[0], php.levelT_[1]);
#ifdef EDM_ML_DEBUG
  int ntot(0);
#endif
  while (dodet) {
#ifdef EDM_ML_DEBUG
    ++ntot;
#endif
    std::vector<int> copy = fv.copyNumbers();
    int nsiz = static_cast<int>(copy.size());
    if (nsiz < levelTop) {
      int lay = copy[nsiz - 1];
      int zside = (nsiz > php.levelZSide_) ? copy[php.levelZSide_] : -1;
      if (zside != 1)
        zside = -1;
      const DDSolid& sol = fv.logicalPart().solid();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << sol.name() << " shape " << sol.shape() << " size " << nsiz << ":" << levelTop
                                    << " lay " << lay << " z " << zside;
#endif
      if (lay == 0) {
        throw cms::Exception("DDException")
            << "Funny layer # " << lay << " zp " << zside << " in " << nsiz << " components";
      } else if (sol.shape() == DDSolidShape::ddtubs) {
        if (std::find(php.layer_.begin(), php.layer_.end(), lay) == php.layer_.end())
          php.layer_.emplace_back(lay);
        const DDTubs& tube = static_cast<DDTubs>(sol);
        double rin = HGCalParameters::k_ScaleFromDDD * tube.rIn();
        double rout = HGCalParameters::k_ScaleFromDDD * tube.rOut();
        auto itr = layers.find(lay);
        if (itr == layers.end()) {
          double zp = HGCalParameters::k_ScaleFromDDD * fv.translation().Z();
          HGCalGeomParameters::layerParameters laypar(rin, rout, zp);
          layers[lay] = laypar;
        } else {
          (itr->second).rmin = std::min(rin, (itr->second).rmin);
          (itr->second).rmax = std::max(rout, (itr->second).rmax);
        }
        if (trforms.find(std::make_pair(lay, zside)) == trforms.end()) {
          DD3Vector x, y, z;
          fv.rotation().GetComponents(x, y, z);
          const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
          const CLHEP::HepRotation hr(rotation);
          double xx =
              ((std::abs(fv.translation().X()) < tolerance) ? 0
                                                            : HGCalParameters::k_ScaleFromDDD * fv.translation().X());
          double yy =
              ((std::abs(fv.translation().Y()) < tolerance) ? 0
                                                            : HGCalParameters::k_ScaleFromDDD * fv.translation().Y());
          const CLHEP::Hep3Vector h3v(xx, yy, HGCalParameters::k_ScaleFromDDD * fv.translation().Z());
          HGCalParameters::hgtrform mytrf;
          mytrf.zp = zside;
          mytrf.lay = lay;
          mytrf.sec = 0;
          mytrf.subsec = 0;
          mytrf.h3v = h3v;
          mytrf.hr = hr;
          trforms[std::make_pair(lay, zside)] = mytrf;
        }
      }
    }
    dodet = fv.next();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Total # of views " << ntot;
#endif
  loadGeometryHexagon8(layers, trforms, firstLayer, php);
}

void HGCalGeomParameters::loadGeometryHexagon8(const cms::DDCompactView* cpv,
                                               HGCalParameters& php,
                                               const std::string& sdTag1,
                                               int firstLayer) {
  const cms::DDFilter filter("Volume", sdTag1);
  cms::DDFilteredView fv((*cpv), filter);
  std::map<int, HGCalGeomParameters::layerParameters> layers;
  std::map<std::pair<int, int>, HGCalParameters::hgtrform> trforms;
  int levelTop = 3 + std::max(php.levelT_[0], php.levelT_[1]);
#ifdef EDM_ML_DEBUG
  int ntot(0);
#endif
  while (fv.firstChild()) {
#ifdef EDM_ML_DEBUG
    ++ntot;
#endif
    // Layers first
    int nsiz = static_cast<int>(fv.level());
    if (nsiz < levelTop) {
      std::vector<int> copy = fv.copyNos();
      int lay = copy[0];
      int zside = (nsiz > php.levelZSide_) ? copy[nsiz - php.levelZSide_ - 1] : -1;
      if (zside != 1)
        zside = -1;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << fv.name() << " shape " << cms::dd::name(cms::DDSolidShapeMap, fv.shape())
                                    << " size " << nsiz << ":" << levelTop << " lay " << lay << " z " << zside << ":"
                                    << php.levelZSide_;
#endif
      if (lay == 0) {
        throw cms::Exception("DDException")
            << "Funny layer # " << lay << " zp " << zside << " in " << nsiz << " components";
      } else if (fv.shape() == cms::DDSolidShape::ddtubs) {
        if (std::find(php.layer_.begin(), php.layer_.end(), lay) == php.layer_.end())
          php.layer_.emplace_back(lay);
        const std::vector<double>& pars = fv.parameters();
        double rin = HGCalParameters::k_ScaleFromDD4hep * pars[0];
        double rout = HGCalParameters::k_ScaleFromDD4hep * pars[1];
        auto itr = layers.find(lay);
        if (itr == layers.end()) {
          double zp = HGCalParameters::k_ScaleFromDD4hep * fv.translation().Z();
          HGCalGeomParameters::layerParameters laypar(rin, rout, zp);
          layers[lay] = laypar;
        } else {
          (itr->second).rmin = std::min(rin, (itr->second).rmin);
          (itr->second).rmax = std::max(rout, (itr->second).rmax);
        }
        if (trforms.find(std::make_pair(lay, zside)) == trforms.end()) {
          DD3Vector x, y, z;
          fv.rotation().GetComponents(x, y, z);
          const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
          const CLHEP::HepRotation hr(rotation);
          double xx = ((std::abs(fv.translation().X()) < tolerance)
                           ? 0
                           : HGCalParameters::k_ScaleFromDD4hep * fv.translation().X());
          double yy = ((std::abs(fv.translation().Y()) < tolerance)
                           ? 0
                           : HGCalParameters::k_ScaleFromDD4hep * fv.translation().Y());
          const CLHEP::Hep3Vector h3v(xx, yy, HGCalParameters::k_ScaleFromDD4hep * fv.translation().Z());
          HGCalParameters::hgtrform mytrf;
          mytrf.zp = zside;
          mytrf.lay = lay;
          mytrf.sec = 0;
          mytrf.subsec = 0;
          mytrf.h3v = h3v;
          mytrf.hr = hr;
          trforms[std::make_pair(lay, zside)] = mytrf;
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Total # of views " << ntot;
#endif
  loadGeometryHexagon8(layers, trforms, firstLayer, php);
}

void HGCalGeomParameters::loadGeometryHexagonModule(const DDCompactView* cpv,
                                                    HGCalParameters& php,
                                                    const std::string& sdTag1,
                                                    const std::string& sdTag2,
                                                    int firstLayer) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters (DDD)::loadGeometryHexagonModule called with tags " << sdTag1
                                << ":" << sdTag2 << " firstLayer " << firstLayer << ":" << php.firstMixedLayer_;
  int ntot1(0), ntot2(0);
#endif
  std::map<int, HGCalGeomParameters::layerParameters> layers;
  std::map<std::pair<int, int>, double> zvals;
  std::map<std::pair<int, int>, HGCalParameters::hgtrform> trforms;
  int levelTop = php.levelT_[0];

  std::string attribute = "Volume";
  DDValue val1(attribute, sdTag2, 0.0);
  DDSpecificsMatchesValueFilter filter1{val1};
  DDFilteredView fv1(*cpv, filter1);
  bool dodet = fv1.firstChild();
  while (dodet) {
#ifdef EDM_ML_DEBUG
    ++ntot1;
#endif
    std::vector<int> copy = fv1.copyNumbers();
    int nsiz = static_cast<int>(copy.size());
    if (levelTop < nsiz) {
      int lay = copy[levelTop];
      int zside = (nsiz > php.levelZSide_) ? copy[php.levelZSide_] : -1;
      if (zside != 1)
        zside = -1;
      if (lay == 0) {
        throw cms::Exception("DDException")
            << "Funny layer # " << lay << " zp " << zside << " in " << nsiz << " components";
      } else {
        if (zvals.find(std::make_pair(lay, zside)) == zvals.end()) {
          zvals[std::make_pair(lay, zside)] = HGCalParameters::k_ScaleFromDDD * fv1.translation().Z();
#ifdef EDM_ML_DEBUG
          std::ostringstream st1;
          st1 << "Name0 " << fv1.name() << " LTop " << levelTop << ":" << lay << " ZSide " << zside << " # of levels "
              << nsiz;
          for (const auto& c : copy)
            st1 << ":" << c;
          st1 << " Z " << zvals[std::make_pair(lay, zside)];
          edm::LogVerbatim("HGCalGeom") << st1.str();
#endif
        }
      }
    }
    dodet = fv1.next();
  }

  DDValue val2(attribute, sdTag1, 0.0);
  DDSpecificsMatchesValueFilter filter2{val2};
  DDFilteredView fv2(*cpv, filter2);
  dodet = fv2.firstChild();
  while (dodet) {
#ifdef EDM_ML_DEBUG
    ++ntot2;
#endif
    std::vector<int> copy = fv2.copyNumbers();
    int nsiz = static_cast<int>(copy.size());
    if (levelTop < nsiz) {
      int lay = copy[levelTop];
      int zside = (nsiz > php.levelZSide_) ? copy[php.levelZSide_] : -1;
      if (zside != 1)
        zside = -1;
      const DDSolid& sol = fv2.logicalPart().solid();
#ifdef EDM_ML_DEBUG
      std::ostringstream st2;
      st2 << "Name1 " << sol.name() << " shape " << sol.shape() << " LTop " << levelTop << ":" << lay << " ZSide "
          << zside << ":" << php.levelZSide_ << " # of levels " << nsiz;
      for (const auto& c : copy)
        st2 << ":" << c;
      edm::LogVerbatim("HGCalGeom") << st2.str();
#endif
      if (lay == 0) {
        throw cms::Exception("DDException")
            << "Funny layer # " << lay << " zp " << zside << " in " << nsiz << " components";
      } else if (sol.shape() == DDSolidShape::ddtubs) {
        if (zvals.find(std::make_pair(lay, zside)) != zvals.end()) {
          if (std::find(php.layer_.begin(), php.layer_.end(), lay) == php.layer_.end())
            php.layer_.emplace_back(lay);
          auto itr = layers.find(lay);
          if (itr == layers.end()) {
            const DDTubs& tube = static_cast<DDTubs>(sol);
            double rin = HGCalParameters::k_ScaleFromDDD * tube.rIn();
            double rout = (php.firstMixedLayer_ > 0 && lay >= php.firstMixedLayer_)
                              ? php.radiusMixBoundary_[lay - php.firstMixedLayer_]
                              : HGCalParameters::k_ScaleFromDDD * tube.rOut();
            double zp = zvals[std::make_pair(lay, 1)];
            HGCalGeomParameters::layerParameters laypar(rin, rout, zp);
            layers[lay] = laypar;
#ifdef EDM_ML_DEBUG
            std::ostringstream st3;
            st3 << "Name1 " << fv2.name() << " LTop " << levelTop << ":" << lay << " ZSide " << zside << " # of levels "
                << nsiz;
            for (const auto& c : copy)
              st3 << ":" << c;
            st3 << " R " << rin << ":" << rout;
            edm::LogVerbatim("HGCalGeom") << st3.str();
#endif
          }

          if (trforms.find(std::make_pair(lay, zside)) == trforms.end()) {
            DD3Vector x, y, z;
            fv2.rotation().GetComponents(x, y, z);
            const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
            const CLHEP::HepRotation hr(rotation);
            double xx = ((std::abs(fv2.translation().X()) < tolerance)
                             ? 0
                             : HGCalParameters::k_ScaleFromDDD * fv2.translation().X());
            double yy = ((std::abs(fv2.translation().Y()) < tolerance)
                             ? 0
                             : HGCalParameters::k_ScaleFromDDD * fv2.translation().Y());
            const CLHEP::Hep3Vector h3v(xx, yy, zvals[std::make_pair(lay, zside)]);
            HGCalParameters::hgtrform mytrf;
            mytrf.zp = zside;
            mytrf.lay = lay;
            mytrf.sec = 0;
            mytrf.subsec = 0;
            mytrf.h3v = h3v;
            mytrf.hr = hr;
            trforms[std::make_pair(lay, zside)] = mytrf;
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "Translation " << h3v;
#endif
          }
        }
      }
    }
    dodet = fv2.next();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Total # of views " << ntot1 << ":" << ntot2;
#endif
  loadGeometryHexagon8(layers, trforms, firstLayer, php);
}

void HGCalGeomParameters::loadGeometryHexagonModule(const cms::DDCompactView* cpv,
                                                    HGCalParameters& php,
                                                    const std::string& sdTag1,
                                                    const std::string& sdTag2,
                                                    int firstLayer) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters (DD4hep)::loadGeometryHexagonModule called with tags " << sdTag1
                                << ":" << sdTag2 << " firstLayer " << firstLayer;
  int ntot1(0), ntot2(0);
#endif
  std::map<int, HGCalGeomParameters::layerParameters> layers;
  std::map<std::pair<int, int>, HGCalParameters::hgtrform> trforms;
  std::map<std::pair<int, int>, double> zvals;
  int levelTop = php.levelT_[0];

  const cms::DDFilter filter1("Volume", sdTag2);
  cms::DDFilteredView fv1((*cpv), filter1);
  while (fv1.firstChild()) {
#ifdef EDM_ML_DEBUG
    ++ntot1;
#endif
    int nsiz = static_cast<int>(fv1.level());
    if (nsiz > levelTop) {
      std::vector<int> copy = fv1.copyNos();
      int lay = copy[nsiz - levelTop - 1];
      int zside = (nsiz > php.levelZSide_) ? copy[nsiz - php.levelZSide_ - 1] : -1;
      if (zside != 1)
        zside = -1;
      if (lay == 0) {
        throw cms::Exception("DDException")
            << "Funny layer # " << lay << " zp " << zside << " in " << nsiz << " components";
      } else {
        if (zvals.find(std::make_pair(lay, zside)) == zvals.end()) {
          zvals[std::make_pair(lay, zside)] = HGCalParameters::k_ScaleFromDD4hep * fv1.translation().Z();
#ifdef EDM_ML_DEBUG
          std::ostringstream st1;
          st1 << "Name0 " << fv1.name() << " LTop " << levelTop << ":" << lay << " ZSide " << zside << " # of levels "
              << nsiz;
          for (const auto& c : copy)
            st1 << ":" << c;
          st1 << " Z " << zvals[std::make_pair(lay, zside)];
          edm::LogVerbatim("HGCalGeom") << st1.str();
#endif
        }
      }
    }
  }

  const cms::DDFilter filter2("Volume", sdTag1);
  cms::DDFilteredView fv2((*cpv), filter2);
  while (fv2.firstChild()) {
    // Layers first
    int nsiz = static_cast<int>(fv2.level());
#ifdef EDM_ML_DEBUG
    ++ntot2;
#endif
    if (nsiz > levelTop) {
      std::vector<int> copy = fv2.copyNos();
      int lay = copy[nsiz - levelTop - 1];
      int zside = (nsiz > php.levelZSide_) ? copy[nsiz - php.levelZSide_ - 1] : -1;
      if (zside != 1)
        zside = -1;
#ifdef EDM_ML_DEBUG
      std::ostringstream st2;
      st2 << "Name1 " << fv2.name() << "Shape " << cms::dd::name(cms::DDSolidShapeMap, fv2.shape()) << " LTop "
          << levelTop << ":" << lay << " ZSide " << zside << ":" << php.levelZSide_ << " # of levels " << nsiz;
      for (const auto& c : copy)
        st2 << ":" << c;
      edm::LogVerbatim("HGCalGeom") << st2.str();
#endif
      if (lay == 0) {
        throw cms::Exception("DDException")
            << "Funny layer # " << lay << " zp " << zside << " in " << nsiz << " components";
      } else {
        if (zvals.find(std::make_pair(lay, zside)) != zvals.end()) {
          if (std::find(php.layer_.begin(), php.layer_.end(), lay) == php.layer_.end())
            php.layer_.emplace_back(lay);
          auto itr = layers.find(lay);
          if (itr == layers.end()) {
            const std::vector<double>& pars = fv2.parameters();
            double rin = HGCalParameters::k_ScaleFromDD4hep * pars[0];
            double rout = (php.firstMixedLayer_ > 0 && lay >= php.firstMixedLayer_)
                              ? php.radiusMixBoundary_[lay - php.firstMixedLayer_]
                              : HGCalParameters::k_ScaleFromDD4hep * pars[1];
            double zp = zvals[std::make_pair(lay, 1)];
            HGCalGeomParameters::layerParameters laypar(rin, rout, zp);
            layers[lay] = laypar;
#ifdef EDM_ML_DEBUG
            std::ostringstream st3;
            st3 << "Name2 " << fv2.name() << " LTop " << levelTop << ":" << lay << " ZSide " << zside << " # of levels "
                << nsiz;
            for (const auto& c : copy)
              st3 << ":" << c;
            st3 << " R " << rin << ":" << rout;
            edm::LogVerbatim("HGCalGeom") << st3.str();
#endif
          }

          if (trforms.find(std::make_pair(lay, zside)) == trforms.end()) {
            DD3Vector x, y, z;
            fv2.rotation().GetComponents(x, y, z);
            const CLHEP::HepRep3x3 rotation(x.X(), y.X(), z.X(), x.Y(), y.Y(), z.Y(), x.Z(), y.Z(), z.Z());
            const CLHEP::HepRotation hr(rotation);
            double xx = ((std::abs(fv2.translation().X()) < tolerance)
                             ? 0
                             : HGCalParameters::k_ScaleFromDD4hep * fv2.translation().X());
            double yy = ((std::abs(fv2.translation().Y()) < tolerance)
                             ? 0
                             : HGCalParameters::k_ScaleFromDD4hep * fv2.translation().Y());
            const CLHEP::Hep3Vector h3v(xx, yy, zvals[std::make_pair(lay, zside)]);
            HGCalParameters::hgtrform mytrf;
            mytrf.zp = zside;
            mytrf.lay = lay;
            mytrf.sec = 0;
            mytrf.subsec = 0;
            mytrf.h3v = h3v;
            mytrf.hr = hr;
            trforms[std::make_pair(lay, zside)] = mytrf;
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "Translation " << h3v;
#endif
          }
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Total # of views " << ntot1 << ":" << ntot2;
#endif
  loadGeometryHexagon8(layers, trforms, firstLayer, php);
}

void HGCalGeomParameters::loadGeometryHexagon8(const std::map<int, HGCalGeomParameters::layerParameters>& layers,
                                               std::map<std::pair<int, int>, HGCalParameters::hgtrform>& trforms,
                                               const int& firstLayer,
                                               HGCalParameters& php) {
  double rmin(0), rmax(0);
  for (unsigned int i = 0; i < layers.size(); ++i) {
    for (auto& layer : layers) {
      if (layer.first == static_cast<int>(i + firstLayer)) {
        php.layerIndex_.emplace_back(i);
        php.rMinLayHex_.emplace_back(layer.second.rmin);
        php.rMaxLayHex_.emplace_back(layer.second.rmax);
        php.zLayerHex_.emplace_back(layer.second.zpos);
        if (i == 0) {
          rmin = layer.second.rmin;
          rmax = layer.second.rmax;
        } else {
          if (rmin > layer.second.rmin)
            rmin = layer.second.rmin;
          if (rmax < layer.second.rmax)
            rmax = layer.second.rmax;
        }
        break;
      }
    }
  }
  php.rLimit_.emplace_back(rmin);
  php.rLimit_.emplace_back(rmax);
  php.depth_ = php.layer_;
  php.depthIndex_ = php.layerIndex_;
  php.depthLayerF_ = php.layerIndex_;

  for (unsigned int i = 0; i < php.layer_.size(); ++i) {
    for (auto& trform : trforms) {
      if (trform.first.first == static_cast<int>(i + firstLayer)) {
        php.fillTrForm(trform.second);
      }
    }
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Minimum/maximum R " << php.rLimit_[0] << ":" << php.rLimit_[1];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters finds " << php.zLayerHex_.size() << " layers";
  for (unsigned int i = 0; i < php.zLayerHex_.size(); ++i) {
    int k = php.layerIndex_[i];
    edm::LogVerbatim("HGCalGeom") << "Layer[" << i << ":" << k << ":" << php.layer_[k]
                                  << "] with r = " << php.rMinLayHex_[i] << ":" << php.rMaxLayHex_[i]
                                  << " at z = " << php.zLayerHex_[i];
  }
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
#endif
}

void HGCalGeomParameters::loadSpecParsHexagon(const DDFilteredView& fv,
                                              HGCalParameters& php,
                                              const DDCompactView* cpv,
                                              const std::string& sdTag1,
                                              const std::string& sdTag2) {
  DDsvalues_type sv(fv.mergedSpecifics());
  php.boundR_ = getDDDArray("RadiusBound", sv, 4);
  rescale(php.boundR_, HGCalParameters::k_ScaleFromDDD);
  php.rLimit_ = getDDDArray("RadiusLimits", sv, 2);
  rescale(php.rLimit_, HGCalParameters::k_ScaleFromDDD);
  php.levelT_ = dbl_to_int(getDDDArray("LevelTop", sv, 0));

  // Grouping of layers
  php.layerGroup_ = dbl_to_int(getDDDArray("GroupingZFine", sv, 0));
  php.layerGroupM_ = dbl_to_int(getDDDArray("GroupingZMid", sv, 0));
  php.layerGroupO_ = dbl_to_int(getDDDArray("GroupingZOut", sv, 0));
  php.slopeMin_ = getDDDArray("Slope", sv, 1);
  const auto& dummy2 = getDDDArray("LayerOffset", sv, 0);
  if (!dummy2.empty())
    php.layerOffset_ = dummy2[0];
  else
    php.layerOffset_ = 0;

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

void HGCalGeomParameters::loadSpecParsHexagon(const cms::DDFilteredView& fv,
                                              HGCalParameters& php,
                                              const std::string& sdTag1,
                                              const std::string& sdTag2,
                                              const std::string& sdTag3,
                                              const std::string& sdTag4) {
  php.boundR_ = fv.get<std::vector<double> >(sdTag4, "RadiusBound");
  rescale(php.boundR_, HGCalParameters::k_ScaleFromDD4hep);
  php.rLimit_ = fv.get<std::vector<double> >(sdTag4, "RadiusLimits");
  rescale(php.rLimit_, HGCalParameters::k_ScaleFromDD4hep);
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
  waferSize_ = dummy[0] * HGCalParameters::k_ScaleFromDD4hepToG4;

  // Cell size
  php.cellSize_ = fv.get<std::vector<double> >(sdTag3, "CellSize");
  rescale(php.cellSize_, HGCalParameters::k_ScaleFromDD4hepToG4);

  // Layer Offset
  const auto& dummy2 = fv.get<std::vector<double> >(sdTag1, "LayerOffset");
  if (!dummy2.empty()) {
    php.layerOffset_ = dummy2[0];
  } else {
    php.layerOffset_ = 0;
  }

  loadSpecParsHexagon(php);
}

void HGCalGeomParameters::loadSpecParsHexagon(const HGCalParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: wafer radius ranges"
                                << " for cell grouping " << php.boundR_[0] << ":" << php.boundR_[1] << ":"
                                << php.boundR_[2] << ":" << php.boundR_[3];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Minimum/maximum R " << php.rLimit_[0] << ":" << php.rLimit_[1];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: LevelTop " << php.levelT_[0];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: minimum slope " << php.slopeMin_[0] << " and layer groupings "
                                << "for the 3 ranges:";
  for (unsigned int k = 0; k < php.layerGroup_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.layerGroup_[k] << ":" << php.layerGroupM_[k] << ":"
                                  << php.layerGroupO_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Wafer Size: " << waferSize_;
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: " << php.cellSize_.size() << " cells of sizes:";
  for (unsigned int k = 0; k < php.cellSize_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << php.cellSize_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: First Layer " << php.firstLayer_ << " and layer offset "
                                << php.layerOffset_;
#endif
}

void HGCalGeomParameters::loadSpecParsHexagon8(const DDFilteredView& fv, HGCalParameters& php) {
  DDsvalues_type sv(fv.mergedSpecifics());
  php.cellThickness_ = getDDDArray("CellThickness", sv, 3);
  rescale(php.cellThickness_, HGCalParameters::k_ScaleFromDDD);
  if ((php.mode_ == HGCalGeometryMode::Hexagon8Module) || (php.mode_ == HGCalGeometryMode::Hexagon8Cassette)) {
    php.waferThickness_ = getDDDArray("WaferThickness", sv, 3);
    rescale(php.waferThickness_, HGCalParameters::k_ScaleFromDDD);
  } else {
    for (unsigned int k = 0; k < php.cellThickness_.size(); ++k)
      php.waferThickness_.emplace_back(php.waferThick_);
  }

  php.radius100to200_ = getDDDArray("Radius100to200", sv, 5);
  php.radius200to300_ = getDDDArray("Radius200to300", sv, 5);

  const auto& dummy = getDDDArray("RadiusCuts", sv, 4);
  php.choiceType_ = static_cast<int>(dummy[0]);
  php.nCornerCut_ = static_cast<int>(dummy[1]);
  php.fracAreaMin_ = dummy[2];
  php.zMinForRad_ = HGCalParameters::k_ScaleFromDDD * dummy[3];

  php.radiusMixBoundary_ = fv.vector("RadiusMixBoundary");
  rescale(php.radiusMixBoundary_, HGCalParameters::k_ScaleFromDDD);

  php.slopeMin_ = getDDDArray("SlopeBottom", sv, 0);
  php.zFrontMin_ = getDDDArray("ZFrontBottom", sv, 0);
  rescale(php.zFrontMin_, HGCalParameters::k_ScaleFromDDD);
  php.rMinFront_ = getDDDArray("RMinFront", sv, 0);
  rescale(php.rMinFront_, HGCalParameters::k_ScaleFromDDD);

  php.slopeTop_ = getDDDArray("SlopeTop", sv, 0);
  php.zFrontTop_ = getDDDArray("ZFrontTop", sv, 0);
  rescale(php.zFrontTop_, HGCalParameters::k_ScaleFromDDD);
  php.rMaxFront_ = getDDDArray("RMaxFront", sv, 0);
  rescale(php.rMaxFront_, HGCalParameters::k_ScaleFromDDD);

  php.zRanges_ = fv.vector("ZRanges");
  rescale(php.zRanges_, HGCalParameters::k_ScaleFromDDD);

  const auto& dummy2 = getDDDArray("LayerOffset", sv, 1);
  php.layerOffset_ = dummy2[0];
  php.layerCenter_ = dbl_to_int(fv.vector("LayerCenter"));

  loadSpecParsHexagon8(php);

  // Read in parameters from Philip's file
  if (php.waferMaskMode_ > 1) {
    std::vector<int> layerType, waferIndex, waferProperties;
    std::vector<double> cassetteShift;
    if (php.waferMaskMode_ == siliconFileEE) {
      waferIndex = dbl_to_int(fv.vector("WaferIndexEE"));
      waferProperties = dbl_to_int(fv.vector("WaferPropertiesEE"));
    } else if (php.waferMaskMode_ == siliconCassetteEE) {
      waferIndex = dbl_to_int(fv.vector("WaferIndexEE"));
      waferProperties = dbl_to_int(fv.vector("WaferPropertiesEE"));
      cassetteShift = fv.vector("CassetteShiftEE");
    } else if (php.waferMaskMode_ == siliconFileHE) {
      waferIndex = dbl_to_int(fv.vector("WaferIndexHE"));
      waferProperties = dbl_to_int(fv.vector("WaferPropertiesHE"));
    } else if (php.waferMaskMode_ == siliconCassetteHE) {
      waferIndex = dbl_to_int(fv.vector("WaferIndexHE"));
      waferProperties = dbl_to_int(fv.vector("WaferPropertiesHE"));
      cassetteShift = fv.vector("CassetteShiftHE");
    }
    if ((php.mode_ == HGCalGeometryMode::Hexagon8Module) || (php.mode_ == HGCalGeometryMode::Hexagon8Cassette)) {
      if ((php.waferMaskMode_ == siliconFileEE) || (php.waferMaskMode_ == siliconCassetteEE)) {
        layerType = dbl_to_int(fv.vector("LayerTypesEE"));
      } else if ((php.waferMaskMode_ == siliconFileHE) || (php.waferMaskMode_ == siliconCassetteHE)) {
        layerType = dbl_to_int(fv.vector("LayerTypesHE"));
      }
    }

    php.cassetteShift_ = cassetteShift;
    rescale(php.cassetteShift_, HGCalParameters::k_ScaleFromDDD);
    loadSpecParsHexagon8(php, layerType, waferIndex, waferProperties);
  }
}

void HGCalGeomParameters::loadSpecParsHexagon8(const cms::DDFilteredView& fv,
                                               const cms::DDVectorsMap& vmap,
                                               HGCalParameters& php,
                                               const std::string& sdTag1) {
  php.cellThickness_ = fv.get<std::vector<double> >(sdTag1, "CellThickness");
  rescale(php.cellThickness_, HGCalParameters::k_ScaleFromDD4hep);
  if ((php.mode_ == HGCalGeometryMode::Hexagon8Module) || (php.mode_ == HGCalGeometryMode::Hexagon8Cassette)) {
    php.waferThickness_ = fv.get<std::vector<double> >(sdTag1, "WaferThickness");
    rescale(php.waferThickness_, HGCalParameters::k_ScaleFromDD4hep);
  } else {
    for (unsigned int k = 0; k < php.cellThickness_.size(); ++k)
      php.waferThickness_.emplace_back(php.waferThick_);
  }

  php.radius100to200_ = fv.get<std::vector<double> >(sdTag1, "Radius100to200");
  php.radius200to300_ = fv.get<std::vector<double> >(sdTag1, "Radius200to300");

  const auto& dummy = fv.get<std::vector<double> >(sdTag1, "RadiusCuts");
  if (dummy.size() > 3) {
    php.choiceType_ = static_cast<int>(dummy[0]);
    php.nCornerCut_ = static_cast<int>(dummy[1]);
    php.fracAreaMin_ = dummy[2];
    php.zMinForRad_ = HGCalParameters::k_ScaleFromDD4hep * dummy[3];
  } else {
    php.choiceType_ = php.nCornerCut_ = php.fracAreaMin_ = php.zMinForRad_ = 0;
  }

  php.slopeMin_ = fv.get<std::vector<double> >(sdTag1, "SlopeBottom");
  php.zFrontMin_ = fv.get<std::vector<double> >(sdTag1, "ZFrontBottom");
  rescale(php.zFrontMin_, HGCalParameters::k_ScaleFromDD4hep);
  php.rMinFront_ = fv.get<std::vector<double> >(sdTag1, "RMinFront");
  rescale(php.rMinFront_, HGCalParameters::k_ScaleFromDD4hep);

  php.slopeTop_ = fv.get<std::vector<double> >(sdTag1, "SlopeTop");
  php.zFrontTop_ = fv.get<std::vector<double> >(sdTag1, "ZFrontTop");
  rescale(php.zFrontTop_, HGCalParameters::k_ScaleFromDD4hep);
  php.rMaxFront_ = fv.get<std::vector<double> >(sdTag1, "RMaxFront");
  rescale(php.rMaxFront_, HGCalParameters::k_ScaleFromDD4hep);
  unsigned int kmax = (php.zFrontTop_.size() - php.slopeTop_.size());
  for (unsigned int k = 0; k < kmax; ++k)
    php.slopeTop_.emplace_back(0);

  const auto& dummy2 = fv.get<std::vector<double> >(sdTag1, "LayerOffset");
  if (!dummy2.empty()) {
    php.layerOffset_ = dummy2[0];
  } else {
    php.layerOffset_ = 0;
  }

  for (auto const& it : vmap) {
    if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "RadiusMixBoundary")) {
      for (const auto& i : it.second)
        php.radiusMixBoundary_.emplace_back(HGCalParameters::k_ScaleFromDD4hep * i);
    } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "ZRanges")) {
      for (const auto& i : it.second)
        php.zRanges_.emplace_back(HGCalParameters::k_ScaleFromDD4hep * i);
    } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "LayerCenter")) {
      for (const auto& i : it.second)
        php.layerCenter_.emplace_back(std::round(i));
    }
  }

  loadSpecParsHexagon8(php);

  // Read in parameters from Philip's file
  if (php.waferMaskMode_ > 1) {
    std::vector<int> layerType, waferIndex, waferProperties;
    std::vector<double> cassetteShift;
    if ((php.waferMaskMode_ == siliconFileEE) || (php.waferMaskMode_ == siliconCassetteEE)) {
      for (auto const& it : vmap) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "WaferIndexEE")) {
          for (const auto& i : it.second)
            waferIndex.emplace_back(std::round(i));
        } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "WaferPropertiesEE")) {
          for (const auto& i : it.second)
            waferProperties.emplace_back(std::round(i));
        }
      }
      if (php.waferMaskMode_ == siliconCassetteEE) {
        for (auto const& it : vmap) {
          if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "CassetteShiftEE")) {
            for (const auto& i : it.second)
              cassetteShift.emplace_back(i);
          }
        }
      }
    } else if ((php.waferMaskMode_ == siliconFileHE) || (php.waferMaskMode_ == siliconCassetteHE)) {
      for (auto const& it : vmap) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "WaferIndexHE")) {
          for (const auto& i : it.second)
            waferIndex.emplace_back(std::round(i));
        } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "WaferPropertiesHE")) {
          for (const auto& i : it.second)
            waferProperties.emplace_back(std::round(i));
        }
      }
      if (php.waferMaskMode_ == siliconCassetteHE) {
        for (auto const& it : vmap) {
          if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "CassetteShiftHE")) {
            for (const auto& i : it.second)
              cassetteShift.emplace_back(i);
          }
        }
      }
    }
    if ((php.mode_ == HGCalGeometryMode::Hexagon8Module) || (php.mode_ == HGCalGeometryMode::Hexagon8Cassette)) {
      if ((php.waferMaskMode_ == siliconFileEE) || (php.waferMaskMode_ == siliconFileHE)) {
        for (auto const& it : vmap) {
          if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "LayerTypesEE")) {
            for (const auto& i : it.second)
              layerType.emplace_back(std::round(i));
          }
        }
      } else if (php.waferMaskMode_ == siliconFileHE) {
        for (auto const& it : vmap) {
          if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "LayerTypesHE")) {
            for (const auto& i : it.second)
              layerType.emplace_back(std::round(i));
          }
        }
      }
    }

    php.cassetteShift_ = cassetteShift;
    rescale(php.cassetteShift_, HGCalParameters::k_ScaleFromDD4hep);
    loadSpecParsHexagon8(php, layerType, waferIndex, waferProperties);
  }
}

void HGCalGeomParameters::loadSpecParsHexagon8(HGCalParameters& php) {
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < php.waferThickness_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: wafer[" << k << "] Thickness " << php.waferThickness_[k];
  for (unsigned int k = 0; k < php.cellThickness_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: cell[" << k << "] Thickness " << php.cellThickness_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Polynomial "
                                << "parameters for 120 to 200 micron "
                                << "transition with" << php.radius100to200_.size() << " elements";
  for (unsigned int k = 0; k < php.radius100to200_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Element [" << k << "] " << php.radius100to200_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Polynomial "
                                << "parameters for 200 to 300 micron "
                                << "transition with " << php.radius200to300_.size() << " elements";
  for (unsigned int k = 0; k < php.radius200to300_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Element [" << k << "] " << php.radius200to300_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Parameters for the"
                                << " transition " << php.choiceType_ << ":" << php.nCornerCut_ << ":"
                                << php.fracAreaMin_ << ":" << php.zMinForRad_;
  for (unsigned int k = 0; k < php.radiusMixBoundary_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Mix[" << k << "] R = " << php.radiusMixBoundary_[k];
  for (unsigned int k = 0; k < php.zFrontMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k << "] Bottom Z = " << php.zFrontMin_[k]
                                  << " Slope = " << php.slopeMin_[k] << " rMax = " << php.rMinFront_[k];
  for (unsigned int k = 0; k < php.zFrontTop_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k << "] Top Z = " << php.zFrontTop_[k]
                                  << " Slope = " << php.slopeTop_[k] << " rMax = " << php.rMaxFront_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Z-Boundary " << php.zRanges_[0] << ":" << php.zRanges_[1] << ":"
                                << php.zRanges_[2] << ":" << php.zRanges_[3];
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: LayerOffset " << php.layerOffset_ << " in array of size "
                                << php.layerCenter_.size();
  for (unsigned int k = 0; k < php.layerCenter_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.layerCenter_[k];
#endif
}

void HGCalGeomParameters::loadSpecParsHexagon8(HGCalParameters& php,
                                               const std::vector<int>& layerType,
                                               const std::vector<int>& waferIndex,
                                               const std::vector<int>& waferProperties) {
  // Store parameters from Philip's file
  for (unsigned int k = 0; k < layerType.size(); ++k) {
    php.layerType_.emplace_back(HGCalTypes::layerType(layerType[k]));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Layer[" << k << "] Type " << layerType[k] << ":" << php.layerType_[k];
#endif
  }
  for (unsigned int k = 0; k < php.layerType_.size(); ++k) {
    double cth = (php.layerType_[k] == HGCalTypes::WaferCenterR) ? cos(php.layerRotation_) : 1.0;
    double sth = (php.layerType_[k] == HGCalTypes::WaferCenterR) ? sin(php.layerRotation_) : 0.0;
    php.layerRotV_.emplace_back(std::make_pair(cth, sth));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Layer[" << k << "] Type " << php.layerType_[k] << " cos|sin(Theta) "
                                  << php.layerRotV_[k].first << ":" << php.layerRotV_[k].second;
#endif
  }
  for (unsigned int k = 0; k < waferIndex.size(); ++k) {
    int partial = HGCalProperty::waferPartial(waferProperties[k]);
    int orient = HGCalWaferMask::getRotation(php.waferZSide_, partial, HGCalProperty::waferOrient(waferProperties[k]));
    php.waferInfoMap_[waferIndex[k]] = HGCalParameters::waferInfo(HGCalProperty::waferThick(waferProperties[k]),
                                                                  partial,
                                                                  orient,
                                                                  HGCalProperty::waferCassette(waferProperties[k]));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "[" << k << ":" << waferIndex[k] << ":"
                                  << HGCalWaferIndex::waferLayer(waferIndex[k]) << ":"
                                  << HGCalWaferIndex::waferU(waferIndex[k]) << ":"
                                  << HGCalWaferIndex::waferV(waferIndex[k]) << "]  Thickness type "
                                  << HGCalProperty::waferThick(waferProperties[k]) << " Partial type " << partial
                                  << " Orientation " << HGCalProperty::waferOrient(waferProperties[k]) << ":" << orient;
#endif
  }
}

void HGCalGeomParameters::loadSpecParsTrapezoid(const DDFilteredView& fv, HGCalParameters& php) {
  DDsvalues_type sv(fv.mergedSpecifics());
  php.radiusMixBoundary_ = fv.vector("RadiusMixBoundary");
  rescale(php.radiusMixBoundary_, HGCalParameters::k_ScaleFromDDD);

  php.nPhiBinBH_ = dbl_to_int(getDDDArray("NPhiBinBH", sv, 0));
  php.layerFrontBH_ = dbl_to_int(getDDDArray("LayerFrontBH", sv, 0));
  php.rMinLayerBH_ = getDDDArray("RMinLayerBH", sv, 0);
  rescale(php.rMinLayerBH_, HGCalParameters::k_ScaleFromDDD);
  php.nCellsFine_ = php.nPhiBinBH_[0];
  php.nCellsCoarse_ = php.nPhiBinBH_[1];
  php.cellSize_.emplace_back(2.0 * M_PI / php.nCellsFine_);
  php.cellSize_.emplace_back(2.0 * M_PI / php.nCellsCoarse_);

  php.slopeMin_ = getDDDArray("SlopeBottom", sv, 0);
  php.zFrontMin_ = getDDDArray("ZFrontBottom", sv, 0);
  rescale(php.zFrontMin_, HGCalParameters::k_ScaleFromDDD);
  php.rMinFront_ = getDDDArray("RMinFront", sv, 0);
  rescale(php.rMinFront_, HGCalParameters::k_ScaleFromDDD);

  php.slopeTop_ = getDDDArray("SlopeTop", sv, 0);
  php.zFrontTop_ = getDDDArray("ZFrontTop", sv, 0);
  rescale(php.zFrontTop_, HGCalParameters::k_ScaleFromDDD);
  php.rMaxFront_ = getDDDArray("RMaxFront", sv, 0);
  rescale(php.rMaxFront_, HGCalParameters::k_ScaleFromDDD);

  php.zRanges_ = fv.vector("ZRanges");
  rescale(php.zRanges_, HGCalParameters::k_ScaleFromDDD);

  // Offsets
  const auto& dummy2 = getDDDArray("LayerOffset", sv, 1);
  php.layerOffset_ = dummy2[0];
  php.layerCenter_ = dbl_to_int(fv.vector("LayerCenter"));

  loadSpecParsTrapezoid(php);

  // tile parameters from Katja's file
  if ((php.waferMaskMode_ == scintillatorFile) || (php.waferMaskMode_ == scintillatorCassette)) {
    std::vector<int> tileIndx, tileProperty;
    std::vector<int> tileHEX1, tileHEX2, tileHEX3, tileHEX4;
    std::vector<double> tileRMin, tileRMax;
    std::vector<int> tileRingMin, tileRingMax;
    std::vector<double> cassetteShift;
    tileIndx = dbl_to_int(fv.vector("TileIndex"));
    tileProperty = dbl_to_int(fv.vector("TileProperty"));
    tileHEX1 = dbl_to_int(fv.vector("TileHEX1"));
    tileHEX2 = dbl_to_int(fv.vector("TileHEX2"));
    tileHEX3 = dbl_to_int(fv.vector("TileHEX3"));
    tileHEX4 = dbl_to_int(fv.vector("TileHEX4"));
    tileRMin = fv.vector("TileRMin");
    tileRMax = fv.vector("TileRMax");
    rescale(tileRMin, HGCalParameters::k_ScaleFromDDD);
    rescale(tileRMax, HGCalParameters::k_ScaleFromDDD);
    tileRingMin = dbl_to_int(fv.vector("TileRingMin"));
    tileRingMax = dbl_to_int(fv.vector("TileRingMax"));
    if (php.waferMaskMode_ == scintillatorCassette) {
      if (php.cassettes_ > 0)
        php.nphiCassette_ = php.nCellsCoarse_ / php.cassettes_;
      cassetteShift = fv.vector("CassetteShiftHE");
      rescale(cassetteShift, HGCalParameters::k_ScaleFromDDD);
    }

    php.cassetteShift_ = cassetteShift;
    loadSpecParsTrapezoid(php,
                          tileIndx,
                          tileProperty,
                          tileHEX1,
                          tileHEX2,
                          tileHEX3,
                          tileHEX4,
                          tileRMin,
                          tileRMax,
                          tileRingMin,
                          tileRingMax);
  }
}

void HGCalGeomParameters::loadSpecParsTrapezoid(const cms::DDFilteredView& fv,
                                                const cms::DDVectorsMap& vmap,
                                                HGCalParameters& php,
                                                const std::string& sdTag1) {
  for (auto const& it : vmap) {
    if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "RadiusMixBoundary")) {
      for (const auto& i : it.second)
        php.radiusMixBoundary_.emplace_back(HGCalParameters::k_ScaleFromDD4hep * i);
    } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "ZRanges")) {
      for (const auto& i : it.second)
        php.zRanges_.emplace_back(HGCalParameters::k_ScaleFromDD4hep * i);
    } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "LayerCenter")) {
      for (const auto& i : it.second)
        php.layerCenter_.emplace_back(std::round(i));
    }
  }

  php.nPhiBinBH_ = dbl_to_int(fv.get<std::vector<double> >(sdTag1, "NPhiBinBH"));
  php.layerFrontBH_ = dbl_to_int(fv.get<std::vector<double> >(sdTag1, "LayerFrontBH"));
  php.rMinLayerBH_ = fv.get<std::vector<double> >(sdTag1, "RMinLayerBH");
  rescale(php.rMinLayerBH_, HGCalParameters::k_ScaleFromDD4hep);
  php.nCellsFine_ = php.nPhiBinBH_[0];
  php.nCellsCoarse_ = php.nPhiBinBH_[1];
  php.cellSize_.emplace_back(2.0 * M_PI / php.nCellsFine_);
  php.cellSize_.emplace_back(2.0 * M_PI / php.nCellsCoarse_);

  php.slopeMin_ = fv.get<std::vector<double> >(sdTag1, "SlopeBottom");
  php.zFrontMin_ = fv.get<std::vector<double> >(sdTag1, "ZFrontBottom");
  rescale(php.zFrontMin_, HGCalParameters::k_ScaleFromDD4hep);
  php.rMinFront_ = fv.get<std::vector<double> >(sdTag1, "RMinFront");
  rescale(php.rMinFront_, HGCalParameters::k_ScaleFromDD4hep);

  php.slopeTop_ = fv.get<std::vector<double> >(sdTag1, "SlopeTop");
  php.zFrontTop_ = fv.get<std::vector<double> >(sdTag1, "ZFrontTop");
  rescale(php.zFrontTop_, HGCalParameters::k_ScaleFromDD4hep);
  php.rMaxFront_ = fv.get<std::vector<double> >(sdTag1, "RMaxFront");
  rescale(php.rMaxFront_, HGCalParameters::k_ScaleFromDD4hep);
  unsigned int kmax = (php.zFrontTop_.size() - php.slopeTop_.size());
  for (unsigned int k = 0; k < kmax; ++k)
    php.slopeTop_.emplace_back(0);

  const auto& dummy2 = fv.get<std::vector<double> >(sdTag1, "LayerOffset");
  php.layerOffset_ = dummy2[0];

  loadSpecParsTrapezoid(php);

  // tile parameters from Katja's file
  if ((php.waferMaskMode_ == scintillatorFile) || (php.waferMaskMode_ == scintillatorCassette)) {
    std::vector<int> tileIndx, tileProperty;
    std::vector<int> tileHEX1, tileHEX2, tileHEX3, tileHEX4;
    std::vector<double> tileRMin, tileRMax;
    std::vector<int> tileRingMin, tileRingMax;
    std::vector<double> cassetteShift;
    for (auto const& it : vmap) {
      if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileIndex")) {
        for (const auto& i : it.second)
          tileIndx.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileProperty")) {
        for (const auto& i : it.second)
          tileProperty.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileHEX1")) {
        for (const auto& i : it.second)
          tileHEX1.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileHEX2")) {
        for (const auto& i : it.second)
          tileHEX2.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileHEX3")) {
        for (const auto& i : it.second)
          tileHEX3.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileHEX4")) {
        for (const auto& i : it.second)
          tileHEX4.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileRMin")) {
        for (const auto& i : it.second)
          tileRMin.emplace_back(HGCalParameters::k_ScaleFromDD4hep * i);
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileRMax")) {
        for (const auto& i : it.second)
          tileRMax.emplace_back(HGCalParameters::k_ScaleFromDD4hep * i);
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileRingMin")) {
        for (const auto& i : it.second)
          tileRingMin.emplace_back(std::round(i));
      } else if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "TileRingMax")) {
        for (const auto& i : it.second)
          tileRingMax.emplace_back(std::round(i));
      }
    }
    if (php.waferMaskMode_ == scintillatorCassette) {
      for (auto const& it : vmap) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), "CassetteShiftHE")) {
          for (const auto& i : it.second)
            cassetteShift.emplace_back(i);
        }
      }
    }

    rescale(cassetteShift, HGCalParameters::k_ScaleFromDD4hep);
    php.cassetteShift_ = cassetteShift;
    loadSpecParsTrapezoid(php,
                          tileIndx,
                          tileProperty,
                          tileHEX1,
                          tileHEX2,
                          tileHEX3,
                          tileHEX4,
                          tileRMin,
                          tileRMax,
                          tileRingMin,
                          tileRingMax);
  }
}

void HGCalGeomParameters::loadSpecParsTrapezoid(HGCalParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters:nCells " << php.nCellsFine_ << ":" << php.nCellsCoarse_
                                << " cellSize: " << php.cellSize_[0] << ":" << php.cellSize_[1];
  for (unsigned int k = 0; k < php.layerFrontBH_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Type[" << k << "] Front Layer = " << php.layerFrontBH_[k]
                                  << " rMin = " << php.rMinLayerBH_[k];
  for (unsigned int k = 0; k < php.radiusMixBoundary_.size(); ++k) {
    edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Mix[" << k << "] R = " << php.radiusMixBoundary_[k]
                                  << " Nphi = " << php.scintCells(k + php.firstLayer_)
                                  << " dPhi = " << php.scintCellSize(k + php.firstLayer_);
  }

  for (unsigned int k = 0; k < php.zFrontMin_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k << "] Bottom Z = " << php.zFrontMin_[k]
                                  << " Slope = " << php.slopeMin_[k] << " rMax = " << php.rMinFront_[k];

  for (unsigned int k = 0; k < php.zFrontTop_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Boundary[" << k << "] Top Z = " << php.zFrontTop_[k]
                                  << " Slope = " << php.slopeTop_[k] << " rMax = " << php.rMaxFront_[k];

  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Z-Boundary " << php.zRanges_[0] << ":" << php.zRanges_[1] << ":"
                                << php.zRanges_[2] << ":" << php.zRanges_[3];

  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: LayerOffset " << php.layerOffset_ << " in array of size "
                                << php.layerCenter_.size();
  for (unsigned int k = 0; k < php.layerCenter_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.layerCenter_[k];
#endif
}

void HGCalGeomParameters::loadSpecParsTrapezoid(HGCalParameters& php,
                                                const std::vector<int>& tileIndx,
                                                const std::vector<int>& tileProperty,
                                                const std::vector<int>& tileHEX1,
                                                const std::vector<int>& tileHEX2,
                                                const std::vector<int>& tileHEX3,
                                                const std::vector<int>& tileHEX4,
                                                const std::vector<double>& tileRMin,
                                                const std::vector<double>& tileRMax,
                                                const std::vector<int>& tileRingMin,
                                                const std::vector<int>& tileRingMax) {
  // tile parameters from Katja's file
  for (unsigned int k = 0; k < tileIndx.size(); ++k) {
    php.tileInfoMap_[tileIndx[k]] = HGCalParameters::tileInfo(HGCalProperty::tileType(tileProperty[k]),
                                                              HGCalProperty::tileSiPM(tileProperty[k]),
                                                              tileHEX1[k],
                                                              tileHEX2[k],
                                                              tileHEX3[k],
                                                              tileHEX4[k]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Tile[" << k << ":" << tileIndx[k] << "] "
                                  << " Type " << HGCalProperty::tileType(tileProperty[k]) << " SiPM "
                                  << HGCalProperty::tileSiPM(tileProperty[k]) << " HEX " << std::hex << tileHEX1[k]
                                  << ":" << tileHEX2[k] << ":" << tileHEX3[k] << ":" << tileHEX4[k] << std::dec;
#endif
  }

  for (unsigned int k = 0; k < tileRMin.size(); ++k) {
    php.tileRingR_.emplace_back(tileRMin[k], tileRMax[k]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "TileRingR[" << k << "] " << tileRMin[k] << ":" << tileRMax[k];
#endif
  }

  for (unsigned k = 0; k < tileRingMin.size(); ++k) {
    php.tileRingRange_.emplace_back(tileRingMin[k], tileRingMax[k]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "TileRingRange[" << k << "] " << tileRingMin[k] << ":" << tileRingMax[k];
#endif
  }
}

void HGCalGeomParameters::loadWaferHexagon(HGCalParameters& php) {
  double waferW(HGCalParameters::k_ScaleFromDDD * waferSize_), rmin(HGCalParameters::k_ScaleFromDDD * php.waferR_);
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
  HGCalParameters::layer_map copiesInLayers(php.layer_.size() + 1);
  HGCalParameters::waferT_map waferTypes(ns2 + 1);
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
                    ((corner.first == static_cast<int>(HGCalParameters::k_CornerSize)) ? php.waferCopy_.size() : -1);
            }
            if ((corner.first > 0) && (corner.first < static_cast<int>(HGCalParameters::k_CornerSize))) {
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

void HGCalGeomParameters::loadWaferHexagon8(HGCalParameters& php) {
  double waferW(php.waferSize_);
  double waferS(php.sensorSeparation_);
  auto wType = std::make_unique<HGCalWaferType>(php.radius100to200_,
                                                php.radius200to300_,
                                                HGCalParameters::k_ScaleToDDD * (waferW + waferS),
                                                HGCalParameters::k_ScaleToDDD * php.zMinForRad_,
                                                php.choiceType_,
                                                php.nCornerCut_,
                                                php.fracAreaMin_);

  double rout(php.rLimit_[1]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Input waferWidth " << waferW << ":" << waferS << " R Max: " << rout;
#endif
  // Clear the vectors
  php.waferCopy_.clear();
  php.waferTypeL_.clear();
  php.waferTypeT_.clear();
  php.waferPosX_.clear();
  php.waferPosY_.clear();
  double r = 0.5 * (waferW + waferS);
  double R = 2.0 * r / sqrt3_;
  double dy = 0.75 * R;
  double r1 = 0.5 * waferW;
  double R1 = 2.0 * r1 / sqrt3_;
  int N = (r == 0) ? 3 : (static_cast<int>(0.5 * rout / r) + 3);
  int ns1 = (2 * N + 1) * (2 * N + 1);
  int ns2 = ns1 * php.zLayerHex_.size();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "wafer " << waferW << ":" << waferS << " r " << r << " dy " << dy << " N " << N
                                << " sizes " << ns1 << ":" << ns2;
  std::vector<int> indtypes(ns1 + 1);
  indtypes.clear();
#endif
  HGCalParameters::wafer_map wafersInLayers(ns1 + 1);
  HGCalParameters::wafer_map typesInLayers(ns2 + 1);
  HGCalParameters::waferT_map waferTypes(ns2 + 1);
  int ipos(0), lpos(0), uvmax(0), nwarn(0);
  std::vector<int> uvmx(php.zLayerHex_.size(), 0);
  for (int v = -N; v <= N; ++v) {
    for (int u = -N; u <= N; ++u) {
      int nr = 2 * v;
      int nc = -2 * u + v;
      double xpos = nc * r;
      double ypos = nr * dy;
      int indx = HGCalWaferIndex::waferIndex(0, u, v);
      php.waferCopy_.emplace_back(indx);
      php.waferPosX_.emplace_back(xpos);
      php.waferPosY_.emplace_back(ypos);
      wafersInLayers[indx] = ipos;
      ++ipos;
      std::pair<int, int> corner = HGCalGeomTools::waferCorner(xpos, ypos, r1, R1, 0, rout, false);
      if ((corner.first == static_cast<int>(HGCalParameters::k_CornerSize)) ||
          ((corner.first > 0) && php.defineFull_)) {
        uvmax = std::max(uvmax, std::max(std::abs(u), std::abs(v)));
      }
      for (unsigned int i = 0; i < php.zLayerHex_.size(); ++i) {
        int copy = i + php.layerOffset_;
        std::pair<double, double> xyoff = geomTools_.shiftXY(php.layerCenter_[copy], (waferW + waferS));
        int lay = php.layer_[php.layerIndex_[i]];
        double xpos0 = xpos + xyoff.first;
        double ypos0 = ypos + xyoff.second;
        double zpos = php.zLayerHex_[i];
        int kndx = HGCalWaferIndex::waferIndex(lay, u, v);
        int type(-1);
        if ((php.mode_ == HGCalGeometryMode::Hexagon8File) || (php.mode_ == HGCalGeometryMode::Hexagon8Module) ||
            (php.mode_ == HGCalGeometryMode::Hexagon8Cassette))
          type = wType->getType(kndx, php.waferInfoMap_);
        if (type < 0)
          type = wType->getType(HGCalParameters::k_ScaleToDDD * xpos0,
                                HGCalParameters::k_ScaleToDDD * ypos0,
                                HGCalParameters::k_ScaleToDDD * zpos);
        php.waferTypeL_.emplace_back(type);
        typesInLayers[kndx] = lpos;
        ++lpos;
#ifdef EDM_ML_DEBUG
        indtypes.emplace_back(kndx);
#endif
        std::pair<int, int> corner =
            HGCalGeomTools::waferCorner(xpos0, ypos0, r1, R1, php.rMinLayHex_[i], php.rMaxLayHex_[i], false);
#ifdef EDM_ML_DEBUG
        if (((corner.first == 0) && std::abs(u) < 5 && std::abs(v) < 5) || (std::abs(u) < 2 && std::abs(v) < 2)) {
          edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " R " << php.rMinLayHex_[i] << ":" << php.rMaxLayHex_[i]
                                        << " u " << u << " v " << v << " with " << corner.first << " corners";
        }
#endif
        if ((corner.first == static_cast<int>(HGCalParameters::k_CornerSize)) ||
            ((corner.first > 0) && php.defineFull_)) {
          uvmx[i] = std::max(uvmx[i], std::max(std::abs(u), std::abs(v)));
        }
        if ((corner.first < static_cast<int>(HGCalParameters::k_CornerSize)) && (corner.first > 0)) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " u|v " << u << ":" << v << " with corner "
                                        << corner.first << ":" << corner.second;
#endif
          int wl = HGCalWaferIndex::waferIndex(lay, u, v);
          if (php.waferMaskMode_ > 0) {
            std::pair<int, int> corner0 = HGCalWaferMask::getTypeMode(
                xpos0, ypos0, r1, R1, php.rMinLayHex_[i], php.rMaxLayHex_[i], type, php.waferMaskMode_);
            if ((php.mode_ == HGCalGeometryMode::Hexagon8File) || (php.mode_ == HGCalGeometryMode::Hexagon8Module) ||
                (php.mode_ == HGCalGeometryMode::Hexagon8Cassette)) {
              auto itr = php.waferInfoMap_.find(wl);
              if (itr != php.waferInfoMap_.end()) {
                int part = (itr->second).part;
                int orient = (itr->second).orient;
                bool ok = (php.mode_ == HGCalGeometryMode::Hexagon8Cassette)
                              ? true
                              : HGCalWaferMask::goodTypeMode(
                                    xpos0, ypos0, r1, R1, php.rMinLayHex_[i], php.rMaxLayHex_[i], part, orient, false);
                if (ok)
                  corner0 = std::make_pair(part, (HGCalTypes::k_OffsetRotation + orient));
#ifdef EDM_ML_DEBUG
                edm::LogVerbatim("HGCalGeom")
                    << "Layer:u:v " << i << ":" << lay << ":" << u << ":" << v << " Part " << corner0.first << ":"
                    << part << " Orient " << corner0.second << ":" << orient << " Position " << xpos0 << ":" << ypos0
                    << " delta " << r1 << ":" << R1 << " Limit " << php.rMinLayHex_[i] << ":" << php.rMaxLayHex_[i]
                    << " Compatibiliety Flag " << ok;
#endif
                if (!ok)
                  ++nwarn;
              }
            }
            waferTypes[wl] = corner0;
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "Layer " << lay << " u|v " << u << ":" << v << " Index " << std::hex << wl << std::dec << " pos "
                << xpos0 << ":" << ypos0 << " R " << r1 << ":" << R1 << " Range " << php.rMinLayHex_[i] << ":"
                << php.rMaxLayHex_[i] << type << ":" << php.waferMaskMode_ << " corner " << corner.first << ":"
                << corner.second << " croner0 " << corner0.first << ":" << corner0.second;
#endif
          } else {
            waferTypes[wl] = corner;
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom") << "Layer " << lay << " u|v " << u << ":" << v << " with corner "
                                          << corner.first << ":" << corner.second;
#endif
          }
        }
      }
    }
  }
  if (nwarn > 0)
    edm::LogWarning("HGCalGeom") << "HGCalGeomParameters::loadWafer8: there are " << nwarn
                                 << " wafers with non-matching partial- orientation types";
  php.waferUVMax_ = uvmax;
  php.waferUVMaxLayer_ = uvmx;
  php.wafersInLayers_ = wafersInLayers;
  php.typesInLayers_ = typesInLayers;
  php.waferTypes_ = waferTypes;
  php.nSectors_ = static_cast<int>(php.waferCopy_.size());
  HGCalParameters::hgtrap mytr;
  mytr.lay = 1;
  mytr.bl = php.waferR_;
  mytr.tl = php.waferR_;
  mytr.h = php.waferR_;
  mytr.alpha = 0.0;
  mytr.cellSize = HGCalParameters::k_ScaleToDDD * php.waferSize_;
  for (auto const& dz : php.cellThickness_) {
    mytr.dz = 0.5 * HGCalParameters::k_ScaleToDDD * dz;
    php.fillModule(mytr, false);
  }
  for (unsigned k = 0; k < php.cellThickness_.size(); ++k) {
    HGCalParameters::hgtrap mytr = php.getModule(k, false);
    mytr.bl *= HGCalParameters::k_ScaleFromDDD;
    mytr.tl *= HGCalParameters::k_ScaleFromDDD;
    mytr.h *= HGCalParameters::k_ScaleFromDDD;
    mytr.dz *= HGCalParameters::k_ScaleFromDDD;
    mytr.cellSize *= HGCalParameters::k_ScaleFromDDD;
    php.fillModule(mytr, true);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeomParameters: Total of " << php.waferCopy_.size() << " wafers";
  for (unsigned int k = 0; k < php.waferCopy_.size(); ++k) {
    int id = php.waferCopy_[k];
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << std::hex << id << std::dec << ":"
                                  << HGCalWaferIndex::waferLayer(id) << ":" << HGCalWaferIndex::waferU(id) << ":"
                                  << HGCalWaferIndex::waferV(id) << " x " << php.waferPosX_[k] << " y "
                                  << php.waferPosY_[k] << " index " << php.wafersInLayers_[id];
  }
  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: Total of " << php.waferTypeL_.size() << " wafer types";
  for (unsigned int k = 0; k < php.waferTypeL_.size(); ++k) {
    int id = indtypes[k];
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << php.typesInLayers_[id] << ":" << php.waferTypeL_[k] << " ID "
                                  << std::hex << id << std::dec << ":" << HGCalWaferIndex::waferLayer(id) << ":"
                                  << HGCalWaferIndex::waferU(id) << ":" << HGCalWaferIndex::waferV(id);
  }
#endif

  //Wafer offset
  php.xLayerHex_.clear();
  php.yLayerHex_.clear();
  double waferSize = php.waferSize_ + php.sensorSeparation_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "WaferSize " << waferSize;
#endif
  for (unsigned int k = 0; k < php.zLayerHex_.size(); ++k) {
    int copy = k + php.layerOffset_;
    std::pair<double, double> xyoff = geomTools_.shiftXY(php.layerCenter_[copy], waferSize);
    php.xLayerHex_.emplace_back(xyoff.first);
    php.yLayerHex_.emplace_back(xyoff.second);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Layer[" << k << "] Off " << copy << ":" << php.layerCenter_[copy] << " Shift "
                                  << xyoff.first << ":" << xyoff.second;
#endif
  }
}

void HGCalGeomParameters::loadCellParsHexagon(const DDCompactView* cpv, HGCalParameters& php) {
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

void HGCalGeomParameters::loadCellParsHexagon(const cms::DDVectorsMap& vmap, HGCalParameters& php) {
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

void HGCalGeomParameters::loadCellParsHexagon(const HGCalParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalLoadCellPars: " << php.cellFine_.size() << " rows for fine cells";
  for (unsigned int k = 0; k < php.cellFine_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "]: " << php.cellFine_[k];
  edm::LogVerbatim("HGCalGeom") << "HGCalLoadCellPars: " << php.cellCoarse_.size() << " rows for coarse cells";
  for (unsigned int k = 0; k < php.cellCoarse_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "]: " << php.cellCoarse_[k];
#endif
}

void HGCalGeomParameters::loadCellTrapezoid(HGCalParameters& php) {
  php.xLayerHex_.resize(php.zLayerHex_.size(), 0);
  php.yLayerHex_.resize(php.zLayerHex_.size(), 0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParameters: x|y|zLayerHex in array of size " << php.zLayerHex_.size();
  for (unsigned int k = 0; k < php.zLayerHex_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Layer[" << k << "] Shift " << php.xLayerHex_[k] << ":" << php.yLayerHex_[k] << ":"
                                  << php.zLayerHex_[k];
#endif
  // Find the radius of each eta-partitions

  if ((php.mode_ == HGCalGeometryMode::TrapezoidFile) || (php.mode_ == HGCalGeometryMode::TrapezoidModule) ||
      (php.mode_ == HGCalGeometryMode::TrapezoidCassette)) {
    //Ring radii for each partition
    for (unsigned int k = 0; k < 2; ++k) {
      for (unsigned int kk = 0; kk < php.tileRingR_.size(); ++kk) {
        php.radiusLayer_[k].emplace_back(php.tileRingR_[kk].first);
#ifdef EDM_ML_DEBUG
        double zv = ((k == 0) ? (php.zLayerHex_[php.layerFrontBH_[1] - php.firstLayer_])
                              : (php.zLayerHex_[php.zLayerHex_.size() - 1]));
        double rv = php.radiusLayer_[k].back();
        double eta = -(std::log(std::tan(0.5 * std::atan(rv / zv))));
        edm::LogVerbatim("HGCalGeom") << "New [" << kk << "] new R = " << rv << " Eta = " << eta;
#endif
      }
      php.radiusLayer_[k].emplace_back(php.tileRingR_[php.tileRingR_.size() - 1].second);
    }
    // Minimum and maximum radius index for each layer
    for (unsigned int k = 0; k < php.zLayerHex_.size(); ++k) {
      php.iradMinBH_.emplace_back(1 + php.tileRingRange_[k].first);
      php.iradMaxBH_.emplace_back(1 + php.tileRingRange_[k].second);
#ifdef EDM_ML_DEBUG
      int kk = php.scintType(php.firstLayer_ + static_cast<int>(k));
      edm::LogVerbatim("HGCalGeom") << "New Layer " << k << " Type " << kk << " Low edge " << php.iradMinBH_.back()
                                    << " Top edge " << php.iradMaxBH_.back();
#endif
    }
  } else {
    //Ring radii for each partition
    for (unsigned int k = 0; k < 2; ++k) {
      double rmax = ((k == 0) ? (php.rMaxLayHex_[php.layerFrontBH_[1] - php.firstLayer_] - 1)
                              : (php.rMaxLayHex_[php.rMaxLayHex_.size() - 1]));
      double rv = php.rMinLayerBH_[k];
      double zv = ((k == 0) ? (php.zLayerHex_[php.layerFrontBH_[1] - php.firstLayer_])
                            : (php.zLayerHex_[php.zLayerHex_.size() - 1]));
      php.radiusLayer_[k].emplace_back(rv);
#ifdef EDM_ML_DEBUG
      double eta = -(std::log(std::tan(0.5 * std::atan(rv / zv))));
      edm::LogVerbatim("HGCalGeom") << "Old [" << k << "] rmax " << rmax << " Z = " << zv
                                    << " dEta = " << php.cellSize_[k] << "\nOld[0] new R = " << rv << " Eta = " << eta;
      int kount(1);
#endif
      while (rv < rmax) {
        double eta = -(php.cellSize_[k] + std::log(std::tan(0.5 * std::atan(rv / zv))));
        rv = zv * std::tan(2.0 * std::atan(std::exp(-eta)));
        php.radiusLayer_[k].emplace_back(rv);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Old [" << kount << "] new R = " << rv << " Eta = " << eta;
        ++kount;
#endif
      }
    }
    // Find minimum and maximum radius index for each layer
    for (unsigned int k = 0; k < php.zLayerHex_.size(); ++k) {
      int kk = php.scintType(php.firstLayer_ + static_cast<int>(k));
      std::vector<double>::iterator low, high;
      low = std::lower_bound(php.radiusLayer_[kk].begin(), php.radiusLayer_[kk].end(), php.rMinLayHex_[k]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Old [" << k << "] RLow = " << php.rMinLayHex_[k] << " pos "
                                    << static_cast<int>(low - php.radiusLayer_[kk].begin());
#endif
      if (low == php.radiusLayer_[kk].begin())
        ++low;
      int irlow = static_cast<int>(low - php.radiusLayer_[kk].begin());
      double drlow = php.radiusLayer_[kk][irlow] - php.rMinLayHex_[k];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "irlow " << irlow << " dr " << drlow << " min " << php.minTileSize_;
#endif
      if (drlow < php.minTileSize_) {
        ++irlow;
#ifdef EDM_ML_DEBUG
        drlow = php.radiusLayer_[kk][irlow] - php.rMinLayHex_[k];
        edm::LogVerbatim("HGCalGeom") << "Modified irlow " << irlow << " dr " << drlow;
#endif
      }
      high = std::lower_bound(php.radiusLayer_[kk].begin(), php.radiusLayer_[kk].end(), php.rMaxLayHex_[k]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Old [" << k << "] RHigh = " << php.rMaxLayHex_[k] << " pos "
                                    << static_cast<int>(high - php.radiusLayer_[kk].begin());
#endif
      if (high == php.radiusLayer_[kk].end())
        --high;
      int irhigh = static_cast<int>(high - php.radiusLayer_[kk].begin());
      double drhigh = php.rMaxLayHex_[k] - php.radiusLayer_[kk][irhigh - 1];
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "irhigh " << irhigh << " dr " << drhigh << " min " << php.minTileSize_;
#endif
      if (drhigh < php.minTileSize_) {
        --irhigh;
#ifdef EDM_ML_DEBUG
        drhigh = php.rMaxLayHex_[k] - php.radiusLayer_[kk][irhigh - 1];
        edm::LogVerbatim("HGCalGeom") << "Modified irhigh " << irhigh << " dr " << drhigh;
#endif
      }
      php.iradMinBH_.emplace_back(irlow);
      php.iradMaxBH_.emplace_back(irhigh);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Old Layer " << k << " Type " << kk << " Low edge " << irlow << ":" << drlow
                                    << " Top edge " << irhigh << ":" << drhigh;
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  for (unsigned int k = 0; k < 2; ++k) {
    edm::LogVerbatim("HGCalGeom") << "Type " << k << " with " << php.radiusLayer_[k].size() << " radii";
    for (unsigned int kk = 0; kk < php.radiusLayer_[k].size(); ++kk)
      edm::LogVerbatim("HGCalGeom") << "Ring[" << kk << "] " << php.radiusLayer_[k][kk];
  }
#endif

  // Now define the volumes
  int im(0);
  php.waferUVMax_ = 0;
  HGCalParameters::hgtrap mytr;
  mytr.alpha = 0.0;
  for (unsigned int k = 0; k < php.zLayerHex_.size(); ++k) {
    if (php.iradMaxBH_[k] > php.waferUVMax_)
      php.waferUVMax_ = php.iradMaxBH_[k];
    int kk = ((php.firstLayer_ + static_cast<int>(k)) < php.layerFrontBH_[1]) ? 0 : 1;
    int irm = php.radiusLayer_[kk].size() - 1;
#ifdef EDM_ML_DEBUG
    double rmin = php.radiusLayer_[kk][std::max((php.iradMinBH_[k] - 1), 0)];
    double rmax = php.radiusLayer_[kk][std::min(php.iradMaxBH_[k], irm)];
    edm::LogVerbatim("HGCalGeom") << "Layer " << php.firstLayer_ + k << ":" << kk << " Radius range "
                                  << php.iradMinBH_[k] << ":" << php.iradMaxBH_[k] << ":" << rmin << ":" << rmax;
#endif
    mytr.lay = php.firstLayer_ + k;
    for (int irad = php.iradMinBH_[k]; irad <= php.iradMaxBH_[k]; ++irad) {
      double rmin = php.radiusLayer_[kk][std::max((irad - 1), 0)];
      double rmax = php.radiusLayer_[kk][std::min(irad, irm)];
      mytr.bl = 0.5 * rmin * php.scintCellSize(mytr.lay);
      mytr.tl = 0.5 * rmax * php.scintCellSize(mytr.lay);
      mytr.h = 0.5 * (rmax - rmin);
      mytr.dz = 0.5 * php.waferThick_;
      mytr.cellSize = 0.5 * (rmax + rmin) * php.scintCellSize(mytr.lay);
      php.fillModule(mytr, true);
      mytr.bl *= HGCalParameters::k_ScaleToDDD;
      mytr.tl *= HGCalParameters::k_ScaleToDDD;
      mytr.h *= HGCalParameters::k_ScaleToDDD;
      mytr.dz *= HGCalParameters::k_ScaleToDDD;
      mytr.cellSize *= HGCalParameters::k_ScaleFromDDD;
      php.fillModule(mytr, false);
      if (irad == php.iradMinBH_[k])
        php.firstModule_.emplace_back(im);
      ++im;
      if (irad == php.iradMaxBH_[k] - 1)
        php.lastModule_.emplace_back(im);
    }
  }
  php.nSectors_ = php.waferUVMax_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Maximum radius index " << php.waferUVMax_;
  for (unsigned int k = 0; k < php.firstModule_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Layer " << k + php.firstLayer_ << " Modules " << php.firstModule_[k] << ":"
                                  << php.lastModule_[k];
#endif
}

std::vector<double> HGCalGeomParameters::getDDDArray(const std::string& str, const DDsvalues_type& sv, const int nmin) {
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        throw cms::Exception("DDException")
            << "HGCalGeomParameters:  # of " << str << " bins " << nval << " < " << nmin << " ==> illegal";
      }
    } else {
      if (nval < 1 && nmin == 0) {
        throw cms::Exception("DDException")
            << "HGCalGeomParameters: # of " << str << " bins " << nval << " < 1 ==> illegal"
            << " (nmin=" << nmin << ")";
      }
    }
    return fvec;
  } else {
    if (nmin >= 0) {
      throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
    }
    std::vector<double> fvec;
    return fvec;
  }
}

std::pair<double, double> HGCalGeomParameters::cellPosition(
    const std::vector<HGCalGeomParameters::cellParameters>& wafers,
    std::vector<HGCalGeomParameters::cellParameters>::const_iterator& itrf,
    int wafer,
    double xx,
    double yy) {
  if (itrf == wafers.end()) {
    for (std::vector<HGCalGeomParameters::cellParameters>::const_iterator itr = wafers.begin(); itr != wafers.end();
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

void HGCalGeomParameters::rescale(std::vector<double>& v, const double s) {
  std::for_each(v.begin(), v.end(), [s](double& n) { n *= s; });
}

void HGCalGeomParameters::resetZero(std::vector<double>& v) {
  for (auto& n : v) {
    if (std::abs(n) < tolmin)
      n = 0;
  }
}
