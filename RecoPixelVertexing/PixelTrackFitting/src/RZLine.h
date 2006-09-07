#ifndef PixelTrackFitting_RZLine_H
#define PixelTrackFitting_RZLine_H

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/CommonDetAlgo/interface/GlobalError.h"
#include <vector>

class RZLine {
public:


  RZLine( const std::vector<GlobalPoint> & points, 
          const std::vector<GlobalError> & errors, 
          const std::vector<bool> isBarrel);

  void fit(float & cotTheta, float & intercept, float &covss, float &covii, float &covsi) const; 

  float chi2(float cotTheta, float intercept) const;

private:
  std::vector<float> r,z,errZ;
};
#endif
