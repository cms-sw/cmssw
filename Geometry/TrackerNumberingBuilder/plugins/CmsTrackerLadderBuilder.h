#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLadderBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLadderBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds Pixel Ladders
 */
template <class FilteredView>
class CmsTrackerLadderBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
