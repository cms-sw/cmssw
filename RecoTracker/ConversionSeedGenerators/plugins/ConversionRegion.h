#ifndef CONVERSIONREGION_H
#define CONVERSIONREGION_H

#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class dso_hidden ConversionRegion {

 public:
 ConversionRegion(GlobalPoint& convPoint, GlobalPoint& pvtxPoint, float& cotTheta, double errTheta, int charge):
  _convPoint(convPoint),
    _pvtxPoint(pvtxPoint),
    _cotTheta(cotTheta),
    _errTheta(errTheta),
    _charge(charge)
    {};
    
  ~ConversionRegion(){};

  GlobalPoint convPoint() const {return   _convPoint;}
  GlobalPoint pvtxPoint()const  {return   _pvtxPoint;}
  float cotTheta       () const {return   _cotTheta; }
  double errTheta      () const {return   _errTheta; }
  int  charge          () const {return   _charge;   } 


 private:
  //Data members
  GlobalPoint _convPoint;
  GlobalPoint _pvtxPoint;
  float _cotTheta;
  double _errTheta;
  int _charge;              
};

#endif
