#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include <DD4hep/Filter.h>

bool TrackerParametersFromDD::build(const DDCompactView* cvp, PTrackerParameters& ptp) {
  for (int subdet = 1; subdet <= 6; ++subdet) {
    std::stringstream sstm;
    sstm << "Subdetector" << subdet;
    std::string name = sstm.str();

    auto const& v = cvp->vector(name);
    if (!v.empty()) {
      std::vector<int> subdetPars = dbl_to_int(v);
      putOne(subdet, subdetPars, ptp);
    }
  }

  ptp.vpars = dbl_to_int(cvp->vector("vPars"));

  return true;
}

bool TrackerParametersFromDD::build(const cms::DDCompactView* cvp, PTrackerParameters& ptp) {
  const auto& vmap = cvp->detector()->vectors();
  for (int subdet = 1; subdet <= 6; ++subdet) {
    std::stringstream sstm;
    sstm << "Subdetector" << subdet;
    std::string name = sstm.str();
    for (auto const& it : vmap) {
      if (dd4hep::dd::compareEqual(dd4hep::dd::noNamespace(it.first), name)) {
        std::vector<int> subdetPars;
        for (const auto& i : it.second)
          subdetPars.emplace_back(std::round(i));
        putOne(subdet, subdetPars, ptp);
      }
    }
  }

  // get "vPars" parameter block from XMLs.
  const std::string& vPars = "trackerParameters:vPars";
  for (auto const& parameterXMLBlock : vmap) {
    const std::string& parameterName = parameterXMLBlock.first;
    // Look for vPars parameter XML block.
    if (dd4hep::dd::compareEqual(vPars, parameterName)) {
      const std::vector<double>& parameterValues = parameterXMLBlock.second;
      for (const auto& value : parameterValues) {
        ptp.vpars.emplace_back(std::round(value));
      }
      break;  // Same logic as old DD: it should be found only once.
    }
  }

  return true;
}

void TrackerParametersFromDD::putOne(int subdet, std::vector<int>& vpars, PTrackerParameters& ptp) {
  PTrackerParameters::Item item;
  item.id = subdet;
  item.vpars = vpars;
  ptp.vitems.emplace_back(item);
}
