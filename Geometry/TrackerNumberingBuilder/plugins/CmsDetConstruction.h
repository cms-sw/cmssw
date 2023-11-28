#ifndef Geometry_TrackerNumberingBuilder_CmsDetConstruction_H
#define Geometry_TrackerNumberingBuilder_CmsDetConstruction_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include <string>
#include <vector>

/**
 * Adds GeometricDets representing final modules to the previous level
 */
template <class FilteredView>
class CmsDetConstruction : public CmsTrackerLevelBuilder<FilteredView> {
public:
  void buildComponent(FilteredView&, GeometricDet*, const std::string&) override;

private:
  void buildDets(const FilteredView&, GeometricDet*, const std::string&);
  void buildSmallDetsforGlued(FilteredView&, GeometricDet*, const std::string&);
  void buildSmallDetsforStack(FilteredView&, GeometricDet*, const std::string&);
  void buildSmallDetsfor3D(FilteredView&, GeometricDet*, const std::string&);
};

#endif  // Geometry_TrackerNumberingBuilder_CmsDetConstruction_H
