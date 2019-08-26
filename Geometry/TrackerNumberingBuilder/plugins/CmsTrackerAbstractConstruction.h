#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerAbstractConstruction_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerAbstractConstruction_H

#include <string>

class GeometricDet;

/**
 * Abstract Class to construct a Tracker SubDet
 */
template <class T>
class CmsTrackerAbstractConstruction {
public:
  virtual ~CmsTrackerAbstractConstruction() = default;
  virtual void build(T&, GeometricDet*, std::string) = 0;
};
#endif
