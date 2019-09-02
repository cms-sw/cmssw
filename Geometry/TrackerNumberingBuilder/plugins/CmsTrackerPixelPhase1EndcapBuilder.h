#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase1EndcapBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase1EndcapBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds the pixel phase 1 endcap
 */
template <class T>
class CmsTrackerPixelPhase1EndcapBuilder : public CmsTrackerLevelBuilder<T> {
public:
  CmsTrackerPixelPhase1EndcapBuilder() {}

private:
  void sortNS(T&, GeometricDet*) override;
  void buildComponent(T&, GeometricDet*, std::string) override;
};

#endif
