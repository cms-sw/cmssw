#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPetalBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPetalBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TEC petals
 */
template <class FilteredView>
class CmsTrackerPetalBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
