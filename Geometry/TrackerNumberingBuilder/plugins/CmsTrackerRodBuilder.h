#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerRodBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerRodBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TOB rods
 */
template <class FilteredView>
class CmsTrackerRodBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
