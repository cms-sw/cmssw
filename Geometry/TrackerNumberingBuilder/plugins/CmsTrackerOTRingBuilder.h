#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTRingBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTRingBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs PixelForward Panels. 
 */
template <class FilteredView>
class CmsTrackerOTRingBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
