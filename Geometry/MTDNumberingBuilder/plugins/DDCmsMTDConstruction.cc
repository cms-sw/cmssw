#include "Geometry/MTDNumberingBuilder/plugins/DDCmsMTDConstruction.h"

#include <utility>
#include <sstream>

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDD.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDConstruction.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#ifdef EDM_ML_DEBUG
#include "DataFormats/Math/interface/deltaPhi.h"
using angle_units::operators::convertRadToDeg;
#endif

class DDNameFilter : public DDFilter {
public:
  void addNS(const std::string& addNS) { allowedNS_.emplace_back(addNS); }
  void add(const std::string& add) { allowed_.emplace_back(add); }
  void veto(const std::string& veto) { veto_.emplace_back(veto); }

  bool accept(const DDExpandedView& ev) const final {
    if (allowedNS_.size() == 0 && allowed_.size() == 0 && veto_.size() == 0) {
      return true;
    }
    bool out(false);
    std::string_view currentNSName(ev.logicalPart().name().ns());
    for (const auto& test : allowedNS_) {
      if (currentNSName.find(test) != std::string::npos) {
        out = true;
        if (allowed_.size() > 0 || veto_.size() > 0) {
          std::string_view currentName(ev.logicalPart().name().name());
          for (const auto& test : veto_) {
            if (currentName.find(test) != std::string::npos) {
              return false;
            }
          }
          for (const auto& test : allowed_) {
            if (currentName.find(test) != std::string::npos) {
              return true;
            }
          }
        }
        break;
      }
    }
    return out;
  }

private:
  std::vector<std::string> allowedNS_;
  std::vector<std::string> allowed_;
  std::vector<std::string> veto_;
};

std::unique_ptr<GeometricTimingDet> DDCmsMTDConstruction::construct(const DDCompactView& cpv) {
  std::string attribute{"CMSCutsRegion"};
  DDNameFilter filter;
  filter.addNS("mtd");
  filter.addNS("btl");
  filter.addNS("etl");

  DDFilteredView fv(cpv, filter);

  CmsMTDStringToEnum theCmsMTDStringToEnum;
  // temporary workaround to distinguish BTL scenarios ordering without introducing a dependency on MTDTopologyMode
  auto isBTLV2 = false;
  // temporary workaround to distinguish ETL scenarios ordering without introducing a dependency on MTDTopologyMode
  const bool prev8(fv.name().find("EModule") != std::string::npos);

  // Specify ETL end component
  GeometricTimingDet::GeometricTimingEnumType ETLEndComponent;
  if (prev8) {
    ETLEndComponent = GeometricTimingDet::ETLSensor;
  } else {
    ETLEndComponent = GeometricTimingDet::ETLSensor;
  }

  auto check_root = theCmsMTDStringToEnum.type(ExtractStringFromDD<DDFilteredView>::getString(attribute, &fv));
  if (check_root != GeometricTimingDet::MTD) {
    fv.firstChild();
    auto check_child = theCmsMTDStringToEnum.type(ExtractStringFromDD<DDFilteredView>::getString(attribute, &fv));
    if (check_child != GeometricTimingDet::MTD) {
      throw cms::Exception("Configuration") << " The first child of the DDFilteredView is not what is expected \n"
                                            << ExtractStringFromDD<DDFilteredView>::getString(attribute, &fv) << "\n";
    }
    fv.parent();
  }

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDNumbering") << "Top level node = " << fv.name();
#endif

  auto mtd = std::make_unique<GeometricTimingDet>(&fv, GeometricTimingDet::MTD);
  size_t limit = 0;
  CmsMTDConstruction<DDFilteredView> theCmsMTDConstruction;

  std::vector<GeometricTimingDet*> subdet;
  std::vector<GeometricTimingDet*> layer;

  do {
    GeometricTimingDet::GeometricTimingEnumType fullNode = theCmsMTDStringToEnum.type(fv.name());
    GeometricTimingDet::GeometricTimingEnumType thisNode =
        theCmsMTDStringToEnum.type(fv.name().substr(0, CmsMTDStringToEnum::kModStrLen));
    size_t num = fv.geoHistory().size();

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MTDNumbering") << "Module = " << fv.name() << " fullNode = " << fullNode
                                     << " thisNode = " << thisNode;
#endif

    if (fullNode == GeometricTimingDet::BTL || fullNode == GeometricTimingDet::ETL) {
      limit = 0;

      // define subdetectors as GeometricTimingDet components

      subdet.emplace_back(theCmsMTDConstruction.buildSubdet(fv));
    }
    if (fullNode == GeometricTimingDet::BTLLayer || fullNode == GeometricTimingDet::ETLDisc) {
      layer.emplace_back(theCmsMTDConstruction.buildLayer(fv));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MTDNumbering") << "Number of layers: " << layer.size();
#endif
    }
    //
    // workaround to make old and TDR structure to cohexist until needed
    // the level chosen for old corresponds to wafers for D50 and previous scenarios
    //
    if ((thisNode == GeometricTimingDet::BTLModule) && limit == 0) {
      if (theCmsMTDConstruction.isBTLV2(fv)) {
        limit = num;
        isBTLV2 = true;
      } else {
        limit = num + 1;
      }
    } else if ((thisNode == ETLEndComponent) && limit == 0) {
      limit = num;
    }
    if (num != limit && limit > 0) {
      continue;
    }
    if (thisNode == GeometricTimingDet::BTLModule) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MTDNumbering") << "Registered in GeometricTimingDet as type " << thisNode;
#endif
      theCmsMTDConstruction.buildBTLModule(fv, layer.back());
      limit = num;
    } else if (thisNode == ETLEndComponent) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MTDNumbering") << "Registered in GeometricTimingDet as type " << thisNode;
#endif
      theCmsMTDConstruction.buildETLModule(fv, layer.back());
      limit = num;
    }
  } while (fv.next());

  // sort GeometricTimingDet

#ifdef EDM_ML_DEBUG
  auto comp = mtd->deepComponents();
  std::stringstream before(std::stringstream::in | std::stringstream::out);
  for (const auto& it : comp) {
    before << "ORDER1 " << it->geographicalId().rawId() << " " << it->type() << " " << it->translation().z() << "\n";
  }
  edm::LogVerbatim("MTDNumbering") << "GeometricTimingDet order before sorting \n" << before.str();
#endif

  if (!isBTLV2) {
    for (size_t index = 0; index < layer.size(); index++) {
      GeometricTimingDet::ConstGeometricTimingDetContainer& icomp = layer[index]->components();
      std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderZ);
      std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderRR);
      if (index > 0) {
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderPhi);
      }
    }
  } else {
    for (size_t index = 0; index < layer.size(); index++) {
      GeometricTimingDet::ConstGeometricTimingDetContainer& icomp = layer[index]->components();
      if (index > 0) {
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderZ);
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderRR);
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderPhi);
      } else {
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::btlOrderPhi);
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::btlOrderZ);
      }
    }
  }

  // Add layers to subdetectors:
  // first BTL (one layer only)

  subdet[0]->addComponent(layer[0]);

  // then ETL (number of layers depend on preTDR design or not)

  if (layer.size() == kNLayerPreTDR) {
    subdet[1]->addComponent(layer[1]);
    subdet[2]->addComponent(layer[2]);
  } else if (layer.size() == kNLayerTDR) {
    subdet[1]->addComponent(layer[1]);
    subdet[1]->addComponent(layer[2]);
    subdet[2]->addComponent(layer[3]);
    subdet[2]->addComponent(layer[4]);
  } else {
    throw cms::Exception("MTDNumbering") << "Wrong number of layers: " << layer.size();
  }

  // Add subdetectors to MTD

  mtd.get()->addComponents(subdet);

#ifdef EDM_ML_DEBUG
  comp.clear();
  comp = mtd->deepComponents();
  std::stringstream after(std::stringstream::in | std::stringstream::out);
  for (const auto& it : comp) {
    after << "ORDER2 " << it->geographicalId().rawId() << " " << static_cast<MTDDetId>(it->geographicalId()).mtdRR()
          << " " << it->type() << " " << it->translation().z() << " "
          << convertRadToDeg(angle0to2pi::make0To2pi(it->phi())) << "\n";
  }
  edm::LogVerbatim("MTDNumbering") << "GeometricTimingDet order after sorting \n" << after.str();
#endif

  return mtd;
}

std::unique_ptr<GeometricTimingDet> DDCmsMTDConstruction::construct(const cms::DDCompactView& cpv) {
  cms::DDFilteredView fv(cpv.detector(), cpv.detector()->worldVolume());

  fv.next(0);
  edm::LogVerbatim("DD4hep_MTDNumbering") << fv.path();
  auto mtd = std::make_unique<GeometricTimingDet>(&fv, GeometricTimingDet::MTD);

  cms::DDSpecParRefs ref;
  const cms::DDSpecParRegistry& mypar = cpv.specpars();
  std::string attribute("MtdDDStructure");
  mypar.filter(ref, attribute, "BarrelTimingLayer");
  mypar.filter(ref, attribute, "EndcapTimingLayer");
  fv.mergedSpecifics(ref);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("DD4hep_MTDNumbering") << "Active filters using " << attribute << ":";
  fv.printFilter();
  edm::LogVerbatim("Geometry").log([&ref](auto& log) {
    log << "Filtered DD SpecPar Registry size: " << ref.size() << "\n";
    for (const auto& t : ref) {
      log << "\nSpecPar " << t.first << ":\nRegExps { ";
      for (const auto& ki : t.second->paths)
        log << ki << " ";
      log << "};\n ";
      for (const auto& kl : t.second->spars) {
        log << kl.first << " = ";
        for (const auto& kil : kl.second) {
          log << kil << " ";
        }
        log << "\n ";
      }
    }
  });
#endif

  bool doSubdet = fv.firstChild();
  edm::LogVerbatim("DD4hep_MTDNumbering") << fv.path();

  CmsMTDStringToEnum theCmsMTDStringToEnum;

  CmsMTDConstruction<cms::DDFilteredView> theCmsMTDConstruction;
  // temporary workaround to distinguish BTL scenarios ordering without introducing a dependency on MTDTopologyMode
  auto isBTLV2 = false;
  // temporary workaround to distinguish ETL scenarios ordering without introducing a dependency on MTDTopologyMode
  const bool prev8(fv.name().find("EModule") != std::string::npos);

  // Specify ETL end component
  GeometricTimingDet::GeometricTimingEnumType ETLEndComponent;
  if (prev8) {
    ETLEndComponent = GeometricTimingDet::ETLSensor;
  } else {
    ETLEndComponent = GeometricTimingDet::ETLSensor;
  }

  std::vector<GeometricTimingDet*> subdet;
  std::vector<GeometricTimingDet*> layer;

  while (doSubdet) {
    std::string nodeName(fv.name());
    GeometricTimingDet::GeometricTimingEnumType fullNode = theCmsMTDStringToEnum.type(nodeName);
    GeometricTimingDet::GeometricTimingEnumType thisNode =
        theCmsMTDStringToEnum.type(nodeName.substr(0, CmsMTDStringToEnum::kModStrLen));

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("DD4hep_MTDNumbering") << fv.path();
    edm::LogVerbatim("DD4hep_MTDNumbering")
        << "Module = " << fv.name() << " fullNode = " << fullNode << " thisNode = " << thisNode;
#endif

    if (fullNode == GeometricTimingDet::BTL || fullNode == GeometricTimingDet::ETL) {
      // define subdetectors as GeometricTimingDet components

      subdet.emplace_back(theCmsMTDConstruction.buildSubdet(fv));
    }
    if (fullNode == GeometricTimingDet::BTLLayer || fullNode == GeometricTimingDet::ETLDisc) {
      layer.emplace_back(theCmsMTDConstruction.buildLayer(fv));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DD4hep_MTDNumbering") << "Number of layers: " << layer.size();
#endif
    }
    if (thisNode == GeometricTimingDet::BTLModule) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DD4hep_MTDNumbering") << "Registered in GeometricTimingDet as type " << thisNode;
#endif
      if (isBTLV2 == false) {
        if (theCmsMTDConstruction.isBTLV2(fv)) {
          isBTLV2 = true;
        }
      }
      theCmsMTDConstruction.buildBTLModule(fv, layer.back());
    } else if (thisNode == ETLEndComponent) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("DD4hep_MTDNumbering") << "Registered in GeometricTimingDet as type " << thisNode;
#endif
      theCmsMTDConstruction.buildETLModule(fv, layer.back());
    }

    doSubdet = fv.firstChild();
  }

  // sort GeometricTimingDet

#ifdef EDM_ML_DEBUG
  auto comp = mtd->deepComponents();
  std::stringstream before(std::stringstream::in | std::stringstream::out);
  for (const auto& it : comp) {
    before << "ORDER1 " << it->geographicalId().rawId() << " " << it->type() << " " << it->translation().z() << "\n";
  }
  edm::LogVerbatim("DD4hep_MTDNumbering") << "GeometricTimingDet order before sorting \n" << before.str();
#endif

  if (!isBTLV2) {
    for (size_t index = 0; index < layer.size(); index++) {
      GeometricTimingDet::ConstGeometricTimingDetContainer& icomp = layer[index]->components();
      std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderZ);
      std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderRR);
      if (index > 0) {
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderPhi);
      }
    }
  } else {
    for (size_t index = 0; index < layer.size(); index++) {
      GeometricTimingDet::ConstGeometricTimingDetContainer& icomp = layer[index]->components();
      if (index > 0) {
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderZ);
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderRR);
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderPhi);
      } else {
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::btlOrderPhi);
        std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::btlOrderZ);
      }
    }
  }

  // Add layers to subdetectors:
  // first BTL (one layer only)

  subdet[0]->addComponent(layer[0]);

  // then ETL (number of layers depend on preTDR design or not)

  if (layer.size() == kNLayerPreTDR) {
    subdet[1]->addComponent(layer[1]);
    subdet[2]->addComponent(layer[2]);
  } else if (layer.size() == kNLayerTDR) {
    subdet[1]->addComponent(layer[1]);
    subdet[1]->addComponent(layer[2]);
    subdet[2]->addComponent(layer[3]);
    subdet[2]->addComponent(layer[4]);
  } else {
    throw cms::Exception("DD4hep_MTDNumbering") << "Wrong number of layers: " << layer.size();
  }

  // Add subdetectors to MTD

  mtd.get()->addComponents(subdet);

#ifdef EDM_ML_DEBUG
  comp.clear();
  comp = mtd->deepComponents();
  std::stringstream after(std::stringstream::in | std::stringstream::out);
  for (const auto& it : comp) {
    after << "ORDER2 " << it->geographicalId().rawId() << " " << static_cast<MTDDetId>(it->geographicalId()).mtdRR()
          << " " << it->type() << " " << it->translation().z() << " "
          << convertRadToDeg(angle0to2pi::make0To2pi(it->phi())) << "\n";
  }
  edm::LogVerbatim("DD4hep_MTDNumbering") << "GeometricTimingDet order after sorting \n" << after.str();
#endif

  return mtd;
}
