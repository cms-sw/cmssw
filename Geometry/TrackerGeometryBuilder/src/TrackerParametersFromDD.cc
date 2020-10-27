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

bool TrackerParametersFromDD::build(const cms::DDCompactView* cpv, PTrackerParameters& ptp) {
  const auto& vmap = cpv->detector()->vectors();
  for (int subdet = 1; subdet <= 6; ++subdet) {
    const auto& v = vmap.at("trackerParameters:Subdetector" + std::to_string(subdet));
    std::vector<int> subdetPars;
    std::transform(v.begin(), v.end(), std::back_inserter(subdetPars), [](int i) -> int { return std::round(i); });
    putOne(subdet, subdetPars, ptp);
  }

  // get "vPars" parameter block from XMLs.
  const auto& vPars = vmap.at("trackerParameters:vPars");
  std::transform(vPars.begin(), vPars.end(), std::back_inserter(ptp.vpars), [](int i) -> int { return std::round(i); });

  return true;
}

void TrackerParametersFromDD::putOne(int subdet, std::vector<int>& vpars, PTrackerParameters& ptp) {
  PTrackerParameters::Item item;
  item.id = subdet;
  item.vpars = vpars;
  ptp.vitems.emplace_back(item);
}
