#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerStringBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerStringBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which constructs TIB strings
 */
class CmsTrackerStringBuilder : public CmsTrackerLevelBuilder {
  
 private:
  virtual void sortNS(DDFilteredView& , GeometricDet*);
  virtual void buildComponent(DDFilteredView& , GeometricDet*, std::string);

};

#endif
