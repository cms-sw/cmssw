#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLadderBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLadderBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Class which builds Pixel Ladders
 */
class CmsTrackerLadderBuilder : public CmsTrackerLevelBuilder {
  
 private:
  void sortNS(DDFilteredView& , GeometricDet*) override;
  void buildComponent(DDFilteredView& , GeometricDet*, std::string) override;

};

#endif
