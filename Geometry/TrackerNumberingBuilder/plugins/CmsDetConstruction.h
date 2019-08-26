#ifndef Geometry_TrackerNumberingBuilder_CmsDetConstruction_H
#define Geometry_TrackerNumberingBuilder_CmsDetConstruction_H

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLevelBuilder.h"
#include <string>
#include <vector>

/**
 * Adds GeometricDets representing final modules to the previous level
 */
template <class T>
class CmsDetConstruction : public CmsTrackerLevelBuilder<T> {
public:
  void buildComponent(T&, GeometricDet*, std::string) override;

private:
  void buildDets(const T&, GeometricDet*, std::string);
  void buildSmallDetsforGlued(T&, GeometricDet*, const std::string&);
  void buildSmallDetsforStack(T&, GeometricDet*, const std::string&);
};

#endif  // Geometry_TrackerNumberingBuilder_CmsDetConstruction_H
