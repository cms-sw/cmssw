//#define EDM_ML_DEBUG

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDConstruction.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDD.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"

#include "DataFormats/Math/interface/deltaPhi.h"

using angle_units::operators::convertRadToDeg;

template <class FilteredView>
CmsMTDConstruction<FilteredView>::CmsMTDConstruction() : btlScheme_(), etlScheme_(), baseNumber_() {}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::mtdOrderZ(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  bool order = (a->translation().z() == b->translation().z()) ? a->translation().rho() < b->translation().rho()
                                                              : a->translation().z() < b->translation().z();
  return order;
}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::mtdOrderRR(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  MTDDetId id1(a->geographicalId());
  MTDDetId id2(b->geographicalId());
  return id1.mtdRR() < id2.mtdRR();
}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::mtdOrderPhi(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  MTDDetId id1(a->geographicalId());
  MTDDetId id2(b->geographicalId());
  return (id1.mtdRR() == id2.mtdRR()) && (angle0to2pi::make0To2pi(a->phi()) < angle0to2pi::make0To2pi(b->phi()));
}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::btlOrderPhi(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  return static_cast<int>(convertRadToDeg(angle0to2pi::make0To2pi(a->phi()))) <
         static_cast<int>(convertRadToDeg(angle0to2pi::make0To2pi(b->phi())));
}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::btlOrderZ(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  bool order = (static_cast<int>(convertRadToDeg(angle0to2pi::make0To2pi(a->phi()))) ==
                static_cast<int>(convertRadToDeg(angle0to2pi::make0To2pi(b->phi())))) &&
               (a->translation().z() < b->translation().z());
  return order;
}

template <>
void CmsMTDConstruction<DDFilteredView>::buildBTLModule(DDFilteredView& fv, GeometricTimingDet* mother) {
  std::string nodeName(fv.name());
  GeometricTimingDet* det =
      new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(nodeName.substr(0, CmsMTDStringToEnum::kModStrLen)));

  if (isBTLV2(fv)) {
    auto& gh = fv.geoHistory();

    baseNumber_.reset();
    baseNumber_.setSize(gh.size());

    for (uint i = gh.size(); i-- > 0;) {
      baseNumber_.addLevel(gh[i].logicalPart().name().name(), gh[i].copyno());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CmsMTDConstruction") << gh[i].logicalPart().name().name() << " " << gh[i].copyno();
#endif
    }

    det->setGeographicalID(BTLDetId(btlScheme_.getUnitID(baseNumber_)));

  } else {
    const auto& copyNumbers = fv.copyNumbers();
    auto module_number = copyNumbers[copyNumbers.size() - 2];

    constexpr char positive[] = "PositiveZ";
    constexpr char negative[] = "NegativeZ";

    const std::string& modname(fv.name());
    size_t delim1 = modname.find("BModule");
    size_t delim2 = modname.find("Layer");
    module_number += atoi(modname.substr(delim1 + CmsMTDStringToEnum::kModStrLen, delim2).c_str()) - 1;

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CmsMTDConstruction")
        << "BTLModule = " << modname << " " << copyNumbers[copyNumbers.size() - 3] << " " << module_number;
#endif

    if (modname.find(positive) != std::string::npos) {
      det->setGeographicalID(BTLDetId(1, copyNumbers[copyNumbers.size() - 3], module_number, 0, 1));
    } else if (modname.find(negative) != std::string::npos) {
      det->setGeographicalID(BTLDetId(0, copyNumbers[copyNumbers.size() - 3], module_number, 0, 1));
    } else {
      throw cms::Exception("CmsMTDConstruction::buildBTLModule")
          << "BTL Module " << module_number << " is neither positive nor negative in Z!";
    }
  }

  mother->addComponent(det);
}

template <>
void CmsMTDConstruction<cms::DDFilteredView>::buildBTLModule(cms::DDFilteredView& fv, GeometricTimingDet* mother) {
  std::string nodeName(fv.name());
  GeometricTimingDet* det =
      new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(nodeName.substr(0, CmsMTDStringToEnum::kModStrLen)));

  if (isBTLV2(fv)) {
    baseNumber_.reset();
    baseNumber_.setSize(fv.copyNos().size());

    for (uint i = 0; i < fv.copyNos().size(); i++) {
      std::string_view name((fv.geoHistory()[i])->GetName());
      size_t ipos = name.rfind('_');
      baseNumber_.addLevel(name.substr(0, ipos), fv.copyNos()[i]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CmsMTDConstruction") << name.substr(0, ipos) << " " << fv.copyNos()[i];
#endif
    }

    det->setGeographicalID(BTLDetId(btlScheme_.getUnitID(baseNumber_)));

  } else {
    const auto& copyNumbers = fv.copyNumbers();
    auto module_number = copyNumbers[1];

    constexpr char positive[] = "PositiveZ";
    constexpr char negative[] = "NegativeZ";

    const std::string modname(fv.name());
    size_t delim1 = modname.find("BModule");
    size_t delim2 = modname.find("Layer");
    module_number += atoi(modname.substr(delim1 + CmsMTDStringToEnum::kModStrLen, delim2).c_str()) - 1;

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MTDNumbering") << fv.path() << "\nBTLModule = " << modname << " " << copyNumbers[2] << " "
                                     << module_number;
#endif

    if (modname.find(positive) != std::string::npos) {
      det->setGeographicalID(BTLDetId(1, copyNumbers[2], module_number, 0, 1));
    } else if (modname.find(negative) != std::string::npos) {
      det->setGeographicalID(BTLDetId(0, copyNumbers[2], module_number, 0, 1));
    } else {
      throw cms::Exception("CmsMTDConstruction::buildBTLModule")
          << "BTL Module " << module_number << " is neither positive nor negative in Z!";
    }
  }

  mother->addComponent(det);
}

template <>
void CmsMTDConstruction<DDFilteredView>::buildETLModule(DDFilteredView& fv, GeometricTimingDet* mother) {
  std::string nodeName(fv.name());
  GeometricTimingDet* det =
      new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(nodeName.substr(0, CmsMTDStringToEnum::kModStrLen)));

  if (isETLtdr(fv)) {
    //
    // For the TDR ETL geometry
    // in principle this method works also for the new geometry, if the main loop points to "Timingactive"
    // but backward compatibility is kept in order to avoid change in volume name and number of siblings
    //

    auto& gh = fv.geoHistory();

    baseNumber_.reset();
    baseNumber_.setSize(gh.size());

    for (uint i = gh.size(); i-- > 0;) {
      baseNumber_.addLevel(gh[i].logicalPart().name().name(), gh[i].copyno());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("CmsMTDConstruction") << gh[i].logicalPart().name().name() << " " << gh[i].copyno();
#endif
    }

    det->setGeographicalID(ETLDetId(etlScheme_.getUnitID(baseNumber_)));

  } else {
    const auto& copyNumbers = fv.copyNumbers();
    auto module_number = copyNumbers[copyNumbers.size() - 2];

    size_t delim_ring = det->name().find("EModule");
    size_t delim_disc = det->name().find("Disc");

    std::string ringN = det->name().substr(delim_ring + CmsMTDStringToEnum::kModStrLen, delim_disc);

    const uint32_t side = det->translation().z() > 0 ? 1 : 0;

    // label geographic detid is front or back (even though it is one module per entry here)
    det->setGeographicalID(ETLDetId(side, atoi(ringN.c_str()), module_number, 0));
  }

  mother->addComponent(det);
}

template <>
void CmsMTDConstruction<cms::DDFilteredView>::buildETLModule(cms::DDFilteredView& fv, GeometricTimingDet* mother) {
  std::string nodeName(fv.name());
  GeometricTimingDet* det =
      new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(nodeName.substr(0, CmsMTDStringToEnum::kModStrLen)));

  baseNumber_.reset();
  baseNumber_.setSize(fv.copyNos().size());

  for (uint i = 0; i < fv.copyNos().size(); i++) {
    std::string_view name((fv.geoHistory()[i])->GetName());
    size_t ipos = name.rfind('_');
    baseNumber_.addLevel(name.substr(0, ipos), fv.copyNos()[i]);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("CmsMTDConstruction") << name.substr(0, ipos) << " " << fv.copyNos()[i];
#endif
  }

  det->setGeographicalID(ETLDetId(etlScheme_.getUnitID(baseNumber_)));

  mother->addComponent(det);
}

template <class FilteredView>
GeometricTimingDet* CmsMTDConstruction<FilteredView>::buildSubdet(FilteredView& fv) {
  std::string nodeName(fv.name());
  auto thisDet = theCmsMTDStringToEnum.type(nodeName);
  GeometricTimingDet* subdet = new GeometricTimingDet(&fv, thisDet);

  if (thisDet == GeometricTimingDet::BTL) {
    subdet->setGeographicalID(BTLDetId(0, 0, 0, 0, 0));
  } else if (thisDet == GeometricTimingDet::ETL) {
    const uint32_t side = subdet->translation().z() > 0 ? 1 : 0;
    subdet->setGeographicalID(ETLDetId(side, 0, 0, 0));
  } else {
    throw cms::Exception("CmsMTDConstruction") << " ERROR - I was expecting a SubDet, I got a " << fv.name();
  }

  return subdet;
}

template <class FilteredView>
GeometricTimingDet* CmsMTDConstruction<FilteredView>::buildLayer(FilteredView& fv) {
  std::string nodeName(fv.name());
  auto thisDet = theCmsMTDStringToEnum.type(nodeName);
  GeometricTimingDet* layer = new GeometricTimingDet(&fv, thisDet);

  if (thisDet != GeometricTimingDet::BTLLayer && thisDet != GeometricTimingDet::ETLDisc) {
    throw cms::Exception("CmsMTDConstruction") << " ERROR - I was expecting a SubDet, I got a " << fv.name();
  }

  uint32_t nLayer;
  if (thisDet == GeometricTimingDet::BTLLayer) {
    //
    // only one layer in BTL
    //
    nLayer = 1;
    layer->setGeographicalID(nLayer);
  } else if (thisDet == GeometricTimingDet::ETLDisc) {
    //
    // no change for pre TDR scenarios, otherwise identifiy layer with disc
    //
    nLayer = (fv.name().find("Disc1") != std::string::npos) ? 1 : 2;
    layer->setGeographicalID(nLayer);
  }

  return layer;
}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::isBTLV2(FilteredView& fv) {
  return (fv.name().substr(0, 9) == "BTLModule");
}

template <class FilteredView>
bool CmsMTDConstruction<FilteredView>::isETLtdr(FilteredView& fv) {
  return (fv.name() == "EModule_Timingactive");
}

template class CmsMTDConstruction<DDFilteredView>;
template class CmsMTDConstruction<cms::DDFilteredView>;
