//#define EDM_ML_DEBUG

#include "Geometry/MTDGeometryBuilder/interface/MTDParametersFromDD.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

using namespace MTDTopologyMode;

namespace {
  int getMTDTopologyMode(const char* s, const DDsvalues_type& sv) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string>& fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception("MTDParametersFromDD") << "Failed to get " << s << " tag.";
      }

      int result(-1);
      MTDTopologyMode::Mode eparser = MTDTopologyMode::MTDStringToEnumParser(fvec[0]);
      result = static_cast<int>(eparser);
      return result;
    } else {
      throw cms::Exception("MTDParametersFromDD") << "Failed to get " << s << " tag.";
    }
  }
}  // namespace

bool MTDParametersFromDD::build(const DDCompactView* cvp, PMTDParameters& ptp) {
  std::array<std::string, 2> mtdSubdet{{"BTL", "ETL"}};
  int subdet(0);
  for (const auto& name : mtdSubdet) {
    auto const& v = cvp->vector(name);
    if (!v.empty()) {
      subdet++;
      std::vector<int> subdetPars = dbl_to_int(v);
      putOne(subdet, subdetPars, ptp);
    } else {
      throw cms::Exception("MTDParametersFromDD") << "Not found " << name << " but needed.";
    }
  }

  ptp.vpars_ = dbl_to_int(cvp->vector("vPars"));

  std::string attribute = "OnlyForMTDRecNumbering";
  DDSpecificsHasNamedValueFilter filter1{attribute};
  DDFilteredView fv1(*cvp, filter1);
  bool ok = fv1.firstChild();
  int topoMode(-1);
  if (ok) {
    DDsvalues_type sv(fv1.mergedSpecifics());
    topoMode = getMTDTopologyMode("TopologyMode", sv);
    ptp.topologyMode_ = topoMode;
  } else {
    throw cms::Exception("MTDParametersFromDD") << "Not found " << attribute.c_str() << " but needed.";
  }

  if (MTDTopologyMode::etlLayoutFromTopoMode(topoMode) == ETLDetId::EtlLayout::v5) {
    std::array<std::string, 8> etlLayout{{
        "StartCopyNo_Front_Left",
        "StartCopyNo_Front_Right",
        "StartCopyNo_Back_Left",
        "StartCopyNo_Back_Right",
        "Offset_Front_Left",
        "Offset_Front_Right",
        "Offset_Back_Left",
        "Offset_Back_Right",
    }};
    int sector(10);
    for (const auto& name : etlLayout) {
      auto const& v = cvp->vector(name);
      if (!v.empty()) {
        sector++;
        std::vector<int> ipos = dbl_to_int(v);
        putOne(sector, ipos, ptp);
      } else {
        throw cms::Exception("MTDParametersFromDD") << "Not found " << name << " but needed.";
      }
    }
  }

  return true;
}

bool MTDParametersFromDD::build(const cms::DDCompactView* cvp, PMTDParameters& ptp) {
  cms::DDVectorsMap vmap = cvp->detector()->vectors();

  std::array<std::string, 2> mtdSubdet{{"BTL", "ETL"}};
  int subdet(0);
  for (const auto& name : mtdSubdet) {
    bool found(false);
    for (auto const& it : vmap) {
      if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), name)) {
        subdet++;
        std::vector<int> subdetPars;
        for (const auto& i : it.second)
          subdetPars.emplace_back(std::round(i));
        putOne(subdet, subdetPars, ptp);
        found = true;
        break;
      }
    }
    if (!found) {
      throw cms::Exception("MTDParametersFromDD") << "Not found " << name << " but needed.";
    }
  }

  auto it = vmap.find("vPars");
  if (it != end(vmap)) {
    std::vector<int> tmpVec;
    for (const auto& i : it->second)
      tmpVec.emplace_back(std::round(i));
    ptp.vpars_ = tmpVec;
  }

  cms::DDSpecParRefs ref;
  const cms::DDSpecParRegistry& mypar = cvp->specpars();
  std::string attribute = "OnlyForMTDRecNumbering";
  mypar.filter(ref, attribute, "MTD");

  std::string topoModeS(mypar.specPar("mtdNumbering")->strValue("TopologyMode"));
  int topoMode(-1);
  if (!topoModeS.empty()) {
    MTDTopologyMode::Mode eparser = MTDTopologyMode::MTDStringToEnumParser(topoModeS);
    topoMode = static_cast<int>(eparser);
    ptp.topologyMode_ = topoMode;
  } else {
    throw cms::Exception("MTDParametersFromDD") << "Not found " << attribute.c_str() << " but needed.";
  }

  if (MTDTopologyMode::etlLayoutFromTopoMode(topoMode) == ETLDetId::EtlLayout::v5) {
    std::array<std::string, 8> etlLayout{{
        "StartCopyNo_Front_Left",
        "StartCopyNo_Front_Right",
        "StartCopyNo_Back_Left",
        "StartCopyNo_Back_Right",
        "Offset_Front_Left",
        "Offset_Front_Right",
        "Offset_Back_Left",
        "Offset_Back_Right",
    }};
    int sector(10);  // add vector index with offset, to distinguish from subdet
    for (const auto& name : etlLayout) {
      bool found(false);
      for (auto const& it : vmap) {
        if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), name)) {
          sector++;
          std::vector<int> ipos;
          for (const auto& i : it.second)
            ipos.emplace_back(std::round(i));
          putOne(sector, ipos, ptp);
          found = true;
          break;
        }
      }
      if (!found) {
        throw cms::Exception("MTDParametersFromDD") << "Not found " << name << " but needed.";
      }
    }
  }

  return true;
}

void MTDParametersFromDD::putOne(int subdet, std::vector<int>& vpars, PMTDParameters& ptp) {
  PMTDParameters::Item item;
  item.id_ = subdet;
  item.vpars_ = vpars;
  ptp.vitems_.emplace_back(item);
#ifdef EDM_ML_DEBUG
  auto print_item = [&]() {
    std::stringstream ss;
    ss << item.id_ << " with " << item.vpars_.size() << " elements:";
    for (const auto& thePar : item.vpars_) {
      ss << " " << thePar;
    }
    return ss.str();
  };
  edm::LogInfo("MTDParametersFromDD") << "Adding PMTDParameters item: " << print_item();
#endif
}
