#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase1EndcapBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase1EndcapBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds the pixel phase 1 endcap
 */
template <class FilteredView>
class CmsTrackerPixelPhase1EndcapBuilder : public CmsTrackerLevelBuilder<FilteredView> {
public:
  CmsTrackerPixelPhase1EndcapBuilder() {}

private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
