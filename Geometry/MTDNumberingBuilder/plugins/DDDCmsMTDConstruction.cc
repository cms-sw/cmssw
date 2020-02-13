#include "Geometry/MTDNumberingBuilder/plugins/DDDCmsMTDConstruction.h"

#include <utility>
#include <sstream>

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
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

using namespace cms;

std::unique_ptr<GeometricTimingDet> DDDCmsMTDConstruction::construct(const DDCompactView& cpv) {
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

  DDFilteredView fv(cpv, filter);

  CmsMTDStringToEnum theCmsMTDStringToEnum;

  auto check_root = theCmsMTDStringToEnum.type(ExtractStringFromDDD::getString(attribute, &fv));
  if (check_root != GeometricTimingDet::MTD) {
    fv.firstChild();
    auto check_child = theCmsMTDStringToEnum.type(ExtractStringFromDDD::getString(attribute, &fv));
    if (check_child != GeometricTimingDet::MTD) {
      throw cms::Exception("Configuration") << " The first child of the DDFilteredView is not what is expected \n"
                                            << ExtractStringFromDDD::getString(attribute, &fv) << "\n";
    }
    fv.parent();
  }

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDNumbering") << "Top level node = " << fv.name();
#endif

  auto mtd = std::make_unique<GeometricTimingDet>(&fv, GeometricTimingDet::MTD);
  size_t limit = 0;
  CmsMTDConstruction theCmsMTDConstruction;
  size_t nBTL;
  nBTL = 0;

  do {
    GeometricTimingDet::GeometricTimingEnumType fullNode = theCmsMTDStringToEnum.type(fv.name());
    GeometricTimingDet::GeometricTimingEnumType thisNode = theCmsMTDStringToEnum.type(fv.name().substr(0, 7));
    size_t num = fv.geoHistory().size();

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MTDNumbering") << "Module level = " << limit << " current node level = " << num << " "
                                     << fv.name();
#endif

    if (fullNode == GeometricTimingDet::BTL || fullNode == GeometricTimingDet::ETL) {
      limit = 0;
    }
    //
    // the level chosen corersponds to wafers for D50 and previous scenarios
    //
    if ((thisNode == GeometricTimingDet::BTLModule || thisNode == GeometricTimingDet::ETLModule) && limit == 0) {
      limit = num + 1;
    }
    if (num != limit && limit > 0) {
      continue;
    }
    if (thisNode == GeometricTimingDet::BTLModule) {
      theCmsMTDConstruction.buildBTLModule(fv, mtd.get(), attribute);
      nBTL++;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MTDNumbering") << "Registered in GeometricTimingDet as type " << thisNode;
#endif
      limit = num;
    } else if (thisNode == GeometricTimingDet::ETLModule) {
      theCmsMTDConstruction.buildETLModule(fv, mtd.get(), attribute);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MTDNumbering") << "Registered in GeometricTimingDet as type " << thisNode;
#endif
      limit = num;
    }
  } while (fv.next());

  // sort GeometricTimingDet

  GeometricTimingDet::ConstGeometricTimingDetContainer& comp = mtd->components();

#ifdef EDM_ML_DEBUG
  std::stringstream before(std::stringstream::in | std::stringstream::out);
  for (const auto& it : comp) {
    before << "ORDER1 " << it->geographicalId().rawId() << " " << it->type() << " " << it->translation().z() << "\n";
  }
  edm::LogVerbatim("MTDNumbering") << "GeometricTimingDet order before sorting \n" << before.str();
#endif

  std::stable_sort(comp.begin(), comp.end(), CmsMTDConstruction::mtdOrderZ);
  std::stable_sort(comp.begin(), comp.end(), CmsMTDConstruction::mtdOrderRR);
  std::stable_sort(comp.begin(), comp.end(), CmsMTDConstruction::mtdOrderSubdet);
  GeometricTimingDet::ConstGeometricTimingDetContainer::iterator endbtl =
      std::find(comp.begin(), comp.end(), comp[nBTL]);
  std::stable_sort(endbtl, comp.end(), CmsMTDConstruction::mtdOrderPhi);
  std::stable_sort(endbtl, comp.end(), CmsMTDConstruction::mtdOrderSide);

#ifdef EDM_ML_DEBUG
  std::stringstream after(std::stringstream::in | std::stringstream::out);
  for (const auto& it : comp) {
    after << "ORDER2 " << it->geographicalId().rawId() << " " << it->type() << " " << it->translation().z() << "\n";
  }
  edm::LogVerbatim("MTDNumbering") << "GeometricTimingDet order before sorting \n" << after.str();
#endif

  return mtd;
}
