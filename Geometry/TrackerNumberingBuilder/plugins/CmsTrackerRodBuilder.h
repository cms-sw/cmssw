#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerRodBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerRodBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which constructs TOB rods
 */
class CmsTrackerRodBuilder : public CmsTrackerLevelBuilder {
  
 private:
  void sortNS(DDFilteredView& , GeometricDet*) override;
  void buildComponent(DDFilteredView& , GeometricDet*, std::string) override;

};

#endif
