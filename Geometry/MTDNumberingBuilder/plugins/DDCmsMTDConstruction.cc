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

//#define EDM_ML_DEBUG

class DDNameFilter : public DDFilter {
public:
  void add(const std::string& add) { allowed_.emplace_back(add); }
  void veto(const std::string& veto) { veto_.emplace_back(veto); }

  bool accept(const DDExpandedView& ev) const final {
    for (const auto& test : veto_) {
      if (ev.logicalPart().name().fullname().find(test) != std::string::npos)
        return false;
    }
    for (const auto& test : allowed_) {
      if (ev.logicalPart().name().fullname().find(test) != std::string::npos)
        return true;
    }
    return false;
  }

private:
  std::vector<std::string> allowed_;
  std::vector<std::string> veto_;
};

std::unique_ptr<GeometricTimingDet> DDCmsMTDConstruction::construct(const DDCompactView& cpv) {
  std::string attribute{"CMSCutsRegion"};
  DDNameFilter filter;
  filter.add("mtd:");
  filter.add("btl:");
  filter.add("etl:");
  filter.veto("service");
  filter.veto("support");
  filter.veto("FSide");
  filter.veto("BSide");
  filter.veto("LSide");
  filter.veto("RSide");
  filter.veto("Between");
  filter.veto("SupportPlate");
  filter.veto("Shield");
  filter.veto("ThermalScreen");
  filter.veto("Aluminium_Disc");
  filter.veto("MIC6_Aluminium_Disc");
  filter.veto("ThermalPad");
  filter.veto("AlN");
  filter.veto("LairdFilm");
  filter.veto("ETROC");
  filter.veto("SensorModule");
  filter.veto("DiscSector");
  filter.veto("LGAD_Substrate");

  DDFilteredView fv(cpv, filter);

  CmsMTDStringToEnum theCmsMTDStringToEnum;

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
    // the level chosen corresponds to wafers for D50 and previous scenarios
    //
    if ((thisNode == GeometricTimingDet::BTLModule) && limit == 0) {
      limit = num + 1;
    }
    //
    // workaround to make old and TDR structure to cohexist until needed
    //
    else if ((thisNode == GeometricTimingDet::ETLModule) && limit == 0) {
      if (theCmsMTDConstruction.isETLtdr(fv)) {
        limit = num;
      } else {
        limit = num + 1;
      }
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
    } else if (thisNode == GeometricTimingDet::ETLModule) {
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

  for (size_t index = 0; index < layer.size(); index++) {
    GeometricTimingDet::ConstGeometricTimingDetContainer& icomp = layer[index]->components();
    std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderZ);
    std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderRR);
    if (index > 0) {
      std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<DDFilteredView>::mtdOrderPhi);
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
    after << "ORDER2 " << it->geographicalId().rawId() << " " << it->type() << " " << it->translation().z() << "\n";
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
      theCmsMTDConstruction.buildBTLModule(fv, layer.back());
    } else if (thisNode == GeometricTimingDet::ETLModule) {
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

  for (size_t index = 0; index < layer.size(); index++) {
    GeometricTimingDet::ConstGeometricTimingDetContainer& icomp = layer[index]->components();
    std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<cms::DDFilteredView>::mtdOrderZ);
    std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<cms::DDFilteredView>::mtdOrderRR);
    if (index > 0) {
      std::stable_sort(icomp.begin(), icomp.end(), CmsMTDConstruction<cms::DDFilteredView>::mtdOrderPhi);
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
    after << "ORDER2 " << it->geographicalId().rawId() << " " << it->type() << " " << it->translation().z() << "\n";
  }
  edm::LogVerbatim("DD4hep_MTDNumbering") << "GeometricTimingDet order after sorting \n" << after.str();
#endif

  return mtd;
}
