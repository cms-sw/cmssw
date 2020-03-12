#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPanelBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPanelBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs PixelForward Panels. 
 */
template <class FilteredView>
class CmsTrackerPanelBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
