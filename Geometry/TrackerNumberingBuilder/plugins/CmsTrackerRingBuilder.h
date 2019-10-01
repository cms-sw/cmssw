#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerRingBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerRingBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TID/TEC rings
 */
template <class FilteredView>
class CmsTrackerRingBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
