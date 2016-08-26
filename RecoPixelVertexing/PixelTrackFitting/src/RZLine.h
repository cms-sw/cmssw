#ifndef PixelTrackFitting_RZLine_H
#define PixelTrackFitting_RZLine_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "CommonTools/Utils/interface/DynArray.h"
#include <vector>

class RZLine {
public:


  RZLine( const DynArray<GlobalPoint> & points, 
          const DynArray<GlobalError> & errors, 
          const DynArray<bool>& isBarrel) : RZLine(points.begin(),errors.begin(),isBarrel.begin(),points.size()){}
  RZLine( const GlobalPoint * points,
          const GlobalError * errors,
          const bool * isBarrel, unsigned int size);

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
