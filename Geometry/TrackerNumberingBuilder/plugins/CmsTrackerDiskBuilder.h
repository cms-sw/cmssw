#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDiskBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerDiskBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs PixelForward/Disk.
 */
template <class FilteredView>
class CmsTrackerDiskBuilder : public CmsTrackerLevelBuilder<FilteredView> {
private:
  void sortNS(FilteredView&, GeometricDet*) override;
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;
};

#endif
