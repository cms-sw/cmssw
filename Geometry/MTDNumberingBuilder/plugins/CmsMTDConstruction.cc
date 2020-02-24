#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDConstruction.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/Math/interface/deltaPhi.h"

bool CmsMTDConstruction::mtdOrderZ(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  bool order = (a->translation().z() == b->translation().z()) ? a->translation().rho() < b->translation().rho()
                                                              : a->translation().z() < b->translation().z();
  return order;
}

bool CmsMTDConstruction::mtdOrderRR(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  MTDDetId id1(a->geographicalId());
  MTDDetId id2(b->geographicalId());
  return id1.mtdRR() < id2.mtdRR();
}

bool CmsMTDConstruction::mtdOrderPhi(const GeometricTimingDet* a, const GeometricTimingDet* b) {
  MTDDetId id1(a->geographicalId());
  MTDDetId id2(b->geographicalId());
  return (id1.mtdRR() == id2.mtdRR()) && (angle0to2pi::make0To2pi(a->phi()) < angle0to2pi::make0To2pi(b->phi()));
}

void CmsMTDConstruction::buildBTLModule(DDFilteredView& fv, GeometricTimingDet* mother, const std::string& attribute) {
  GeometricTimingDet* det =
      new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(fv.name().substr(0, CmsMTDStringToEnum::kModStrLen)));

  const auto& copyNumbers = fv.copyNumbers();
  auto module_number = copyNumbers[copyNumbers.size() - 2];

  constexpr char positive[] = "PositiveZ";
  constexpr char negative[] = "NegativeZ";

  const std::string modname = fv.name();
  size_t delim1 = modname.find("BModule");
  size_t delim2 = modname.find("Layer");
  module_number += atoi(modname.substr(delim1 + CmsMTDStringToEnum::kModStrLen, delim2).c_str()) - 1;

  if (modname.find(positive) != std::string::npos) {
    det->setGeographicalID(BTLDetId(1, copyNumbers[copyNumbers.size() - 3], module_number, 0, 1));
  } else if (modname.find(negative) != std::string::npos) {
    det->setGeographicalID(BTLDetId(0, copyNumbers[copyNumbers.size() - 3], module_number, 0, 1));
  } else {
    throw cms::Exception("CmsMTDConstruction::buildBTLModule")
        << "BTL Module " << module_number << " is neither positive nor negative in Z!";
  }

  mother->addComponent(det);
}

void CmsMTDConstruction::buildETLModule(DDFilteredView& fv, GeometricTimingDet* mother, const std::string& attribute) {
  GeometricTimingDet* det =
      new GeometricTimingDet(&fv, theCmsMTDStringToEnum.type(fv.name().substr(0, CmsMTDStringToEnum::kModStrLen)));

  const auto& copyNumbers = fv.copyNumbers();
  auto module_number = copyNumbers[copyNumbers.size() - 2];

  size_t delim_ring = det->name().find("EModule");
  size_t delim_disc = det->name().find("Disc");

  std::string ringN = det->name().substr(delim_ring + CmsMTDStringToEnum::kModStrLen, delim_disc);

  const uint32_t side = det->translation().z() > 0 ? 1 : 0;

  // label geographic detid is front or back (even though it is one module per entry here)
  det->setGeographicalID(ETLDetId(side, atoi(ringN.c_str()), module_number, 0));

  mother->addComponent(det);
}

GeometricTimingDet* CmsMTDConstruction::buildSubdet(DDFilteredView& fv,
                                                    GeometricTimingDet* mother,
                                                    const std::string& attribute) {
  auto thisDet = theCmsMTDStringToEnum.type(fv.name());
  GeometricTimingDet* subdet = new GeometricTimingDet(&fv, thisDet);

  if (thisDet == GeometricTimingDet::BTL) {
    subdet->setGeographicalID(BTLDetId(0, 0, 0, 0, 0));
  } else if (thisDet == GeometricTimingDet::ETL) {
    const uint32_t side = subdet->translation().z() > 0 ? 1 : 0;
    subdet->setGeographicalID(ETLDetId(side, 0, 0, 0));
  } else {
    throw cms::Exception("CmsMTDConstruction") << " ERROR - I was expecting a SubDet, I got a " << fv.name();
  }

  mother->addComponent(subdet);

  return subdet;
}

GeometricTimingDet* CmsMTDConstruction::buildLayer(DDFilteredView& fv,
                                                   GeometricTimingDet* mother,
                                                   const std::string& attribute) {
  auto thisDet = theCmsMTDStringToEnum.type(fv.name());
  GeometricTimingDet* subdet = new GeometricTimingDet(&fv, thisDet);

  if (thisDet != GeometricTimingDet::BTLLayer && thisDet != GeometricTimingDet::ETLDisc) {
    throw cms::Exception("CmsMTDConstruction") << " ERROR - I was expecting a SubDet, I got a " << fv.name();
  }

  mother->addComponent(subdet);

  return subdet;
}
