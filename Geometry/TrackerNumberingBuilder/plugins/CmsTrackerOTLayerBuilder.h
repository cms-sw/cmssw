#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerOTLayerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerOTLayerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which contructs TIB/TOB layers
 */
class CmsTrackerOTLayerBuilder : public CmsTrackerLevelBuilder {
  
 private:
  virtual void sortNS(DDFilteredView& , GeometricDet*);
  virtual void buildComponent(DDFilteredView& , GeometricDet*, std::string);

};

#endif
