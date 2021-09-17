#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerAbstractConstruction_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerAbstractConstruction_H

#include <string>

class GeometricDet;

/**
 * Abstract Class to construct a Tracker SubDet
 */
template <class FilteredView>
class CmsTrackerAbstractConstruction {
public:
  virtual ~CmsTrackerAbstractConstruction() = default;
  virtual void build(FilteredView &, GeometricDet *, const std::string &) = 0;
};
#endif
