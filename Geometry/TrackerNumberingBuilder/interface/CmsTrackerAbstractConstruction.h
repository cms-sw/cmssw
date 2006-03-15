#ifndef Geometry_TrackerNumberingBuilder_CmsTrackerAbstractConstruction_H
#define Geometry_TrackerNumberingBuilder_CmsTrackerAbstractConstruction_H

#include<string>

class GeometricDet;
class DDFilteredView;

/**
 * Abstract Class to construct a Tracker SubDet
 */
class CmsTrackerAbstractConstruction{
 public:
  virtual void build(DDFilteredView& , GeometricDet*, std::string) = 0;

};
#endif
