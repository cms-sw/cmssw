#ifndef BoundingBox_H
#define BoundingBox_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class BoundPlane;

/** A helper class that returns the corners of a rectangle that
 *  fully contains a bound plane.
 */
class BoundingBox {
public:

  BoundingBox(){}
  BoundingBox(const BoundPlane& plane);
  
  // old interface
  static std::vector<GlobalPoint> corners( const BoundPlane&);


  GlobalPoint const & operator[](unsigned int i) const {
    return  m_corners[i];
  }
  GlobalPoint const & corner(unsigned int i) const {
    return  m_corners[i];
  }


private:


  GlobalPoint m_corners[8];

};

#endif
