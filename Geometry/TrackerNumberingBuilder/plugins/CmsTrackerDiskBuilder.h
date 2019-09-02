#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerDiskBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerDiskBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs PixelForward/Disk.
 */
template <class T>
class CmsTrackerDiskBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
