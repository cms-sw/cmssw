#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerPetalBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerPetalBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which constructs TEC petals
 */
class CmsTrackerPetalBuilder : public CmsTrackerLevelBuilder {
  
 private:
  virtual void sortNS(DDFilteredView& , GeometricDet*);
  virtual void buildComponent(DDFilteredView& , GeometricDet*, std::string);

};

#endif
