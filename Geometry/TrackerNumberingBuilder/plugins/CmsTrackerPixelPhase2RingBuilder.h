#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2RingBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPixelPhase2RingBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which contructs PixelForward Panels. 
 */
class CmsTrackerPixelPhase2RingBuilder : public CmsTrackerLevelBuilder {
  
 private:
  virtual void sortNS(DDFilteredView& , GeometricDet*);
  virtual void buildComponent(DDFilteredView& , GeometricDet*, std::string);

};

#endif
