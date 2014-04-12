#ifndef PixelTrackFitting_RZLine_H
#define PixelTrackFitting_RZLine_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include <vector>

class RZLine {
public:


  RZLine( const std::vector<GlobalPoint> & points, 
          const std::vector<GlobalError> & errors, 
          const std::vector<bool>& isBarrel);
  RZLine( const std::vector<float> & aR, 
          const std::vector<float> & aZ, 
          const std::vector<float> & aErrZ);

  void fit(float & cotTheta, float & intercept, float &covss, float &covii, float &covsi) const; 

  float chi2(float cotTheta, float intercept) const;

private:
  std::vector<float> storage;
  int nPoints;
  float *r, *z, *errZ2;
};
#endif
