#ifndef Geometry_MTDNumberingBuilder_CmsMTDAbstractConstruction_H
#define Geometry_MTDNumberingBuilder_CmsMTDAbstractConstruction_H

#include<string>

class GeometricTimingDet;
class DDFilteredView;

/**
 * Abstract Class to construct a Tracker SubDet
 */
class CmsMTDAbstractConstruction{
 public:
  virtual ~CmsMTDAbstractConstruction() = default;
  virtual void build(DDFilteredView& , GeometricTimingDet*, std::string) = 0;

};
#endif
