#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2DiskBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2DiskBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs Phase2 Pixel Tracker/Discs.
 */
template <class T>
class CmsTrackerPixelPhase2DiskBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, const std::string&) override;
};

#endif
