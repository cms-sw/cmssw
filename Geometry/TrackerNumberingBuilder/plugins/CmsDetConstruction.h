#ifndef Geometry_TrackerNumberingBuilder_CmsDetConstruction_H
#define Geometry_TrackerNumberingBuilderCmsDetConstruction_H
#include<string>
#include<vector>
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
/**
 * Adds GeometricDets representing final modules to the previous level
 */
class CmsDetConstruction : public CmsTrackerLevelBuilder {
 public:
  void  buildComponent(DDFilteredView& , GeometricDet*, std::string);
 private:
  void buildDets(DDFilteredView& , GeometricDet* , std::string);
  void buildSmallDetsforGlued(DDFilteredView& , GeometricDet* , std::string);
  void buildSmallDetsforStack(DDFilteredView& , GeometricDet* , std::string);
};



#endif
