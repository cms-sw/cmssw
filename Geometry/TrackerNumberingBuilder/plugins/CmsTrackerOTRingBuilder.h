#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTRingBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTRingBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which contructs PixelForward Panels. 
 */
template <class T>
class CmsTrackerOTRingBuilder : public CmsTrackerLevelBuilder<T> {
private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
