#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerLayerBuilder_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerLayerBuilder_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>
/**
 * Class which contructs TIB/TOB layers
 */
class CmsTrackerLayerBuilder : public CmsTrackerLevelBuilder {
  
 private:
  void sortNS(DDFilteredView& , GeometricDet*) override;
  void buildComponent(DDFilteredView& , GeometricDet*, std::string) override;

};

#endif
