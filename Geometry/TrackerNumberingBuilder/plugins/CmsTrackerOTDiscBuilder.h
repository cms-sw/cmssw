#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTDiscBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTDiscBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs Phase2 Outer Tracker/Discs.
 */
template <class FilteredView>
class CmsTrackerOTDiscBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
