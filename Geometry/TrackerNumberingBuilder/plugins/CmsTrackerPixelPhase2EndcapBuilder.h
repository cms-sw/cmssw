#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2EndcapBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2EndcapBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds the pixel phase 2 endcap
 */
template <class T>
class CmsTrackerPixelPhase2EndcapBuilder : public CmsTrackerLevelBuilder<T> {
public:
  CmsTrackerPixelPhase2EndcapBuilder() {}

private:
  void sortNS( T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
