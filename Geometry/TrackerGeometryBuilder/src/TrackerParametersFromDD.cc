#include "Geometry/TrackerGeometryBuilder/interface/TrackerParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"

bool TrackerParametersFromDD::build(const DDCompactView* cvp, PTrackerParameters& ptp) {
  for (int subdet = 1; subdet <= 6; ++subdet) {
    std::stringstream sstm;
    sstm << "Subdetector" << subdet;
    std::string name = sstm.str();

    if (DDVectorGetter::check(name)) {
      std::vector<int> subdetPars = dbl_to_int(DDVectorGetter::get(name));
      putOne(subdet, subdetPars, ptp);
    }
  }

  ptp.vpars = dbl_to_int(DDVectorGetter::get("vPars"));

  return true;
}

bool TrackerParametersFromDD::build(const cms::DDCompactView* cvp, PTrackerParameters& ptp) {
  cms::DDVectorsMap vmap = cvp->detector()->vectors();
  for (int subdet = 1; subdet <= 6; ++subdet) {
    std::stringstream sstm;
    sstm << "Subdetector" << subdet;
    std::string name = sstm.str();
    for (auto const& it : vmap) {
      if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), name)) {
        std::vector<int> subdetPars;
        for (const auto& i : it.second)
          subdetPars.emplace_back(std::round(i));
        putOne(subdet, subdetPars, ptp);
      }
    }
  }

  auto it = vmap.find("vPars");
  if (it != end(vmap)) {
    std::vector<int> tmpVec;
    for (const auto& i : it->second)
      tmpVec.emplace_back(std::round(i));
    ptp.vpars = tmpVec;
  }

  return true;
}

void TrackerParametersFromDD::putOne(int subdet, std::vector<int>& vpars, PTrackerParameters& ptp) {
  PTrackerParameters::Item item;
  item.id = subdet;
  item.vpars = vpars;
  ptp.vitems.emplace_back(item);
}
