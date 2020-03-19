#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDiskBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPanelBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <algorithm>

using namespace std;

template <class FilteredView>
void CmsTrackerDiskBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsTrackerPanelBuilder<FilteredView> theCmsTrackerPanelBuilder;
  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));

  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::panel:
      theCmsTrackerPanelBuilder.build(fv, subdet, s);
      break;
    default:
      edm::LogError("CmsTrackerDiskBuilder")
          << " ERROR - I was expecting a Panel, I got a " << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }
  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerDiskBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  switch (det->components().front()->type()) {
    case GeometricDet::panel:
      trackerStablePhiSort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::getPhi);
      break;
    default:
      edm::LogError("CmsTrackerDiskBuilder")
          << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();
  }

  GeometricDet::GeometricDetContainer zminpanels;  // Here z refers abs(z);
  GeometricDet::GeometricDetContainer zmaxpanels;  // So, zmin panel is always closer to ip.

  uint32_t totalblade = comp.size() / 2;
  //  std::cout << "pixel_disk " << pixel_disk << endl;

  zminpanels.reserve(totalblade);
  zmaxpanels.reserve(totalblade);
  for (uint32_t j = 0; j < totalblade; j++) {
    if (std::abs(comp[2 * j]->translation().z()) > std::abs(comp[2 * j + 1]->translation().z())) {
      zmaxpanels.emplace_back(det->component(2 * j));
      zminpanels.emplace_back(det->component(2 * j + 1));

    } else if (std::abs(comp[2 * j]->translation().z()) < std::abs(comp[2 * j + 1]->translation().z())) {
      zmaxpanels.emplace_back(det->component(2 * j + 1));
      zminpanels.emplace_back(det->component(2 * j));
    } else {
      edm::LogWarning("CmsTrackerDiskBuilder") << "WARNING - The Z of  both panels are equal! ";
    }
  }

  for (uint32_t fn = 0; fn < zminpanels.size(); fn++) {
    uint32_t blade = fn + 1;
    uint32_t panel = 1;
    uint32_t temp = (blade << 2) | panel;
    zminpanels[fn]->setGeographicalID(temp);
  }

  for (uint32_t bn = 0; bn < zmaxpanels.size(); bn++) {
    uint32_t blade = bn + 1;
    uint32_t panel = 2;
    uint32_t temp = (blade << 2) | panel;
    zmaxpanels[bn]->setGeographicalID(temp);
  }

  det->clearComponents();
  det->addComponents(zminpanels);
  det->addComponents(zmaxpanels);
}

template class CmsTrackerDiskBuilder<DDFilteredView>;
template class CmsTrackerDiskBuilder<cms::DDFilteredView>;
