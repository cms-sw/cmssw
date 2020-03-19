#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerStringBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerStringBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which constructs TIB strings
 */
template <class FilteredView>
class CmsTrackerStringBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
