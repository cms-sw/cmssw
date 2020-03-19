#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerWheelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerWheelBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds TEC wheels
 */
template <class FilteredView>
class CmsTrackerWheelBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
