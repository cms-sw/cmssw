#ifndef ZReflectedMFGrid_H
#define ZReflectedMFGrid_H

#include "MagneticField/Interpolation/interface/MFGrid.h"

class ZReflectedMFGrid : public MFGrid {
public:

  ZReflectedMFGrid( const GloballyPositioned<float>& vol,
		    MFGrid* sectorGrid);

  ~ZReflectedMFGrid();

  virtual LocalVector valueInTesla( const LocalPoint& p) const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const ;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const ;

  virtual Dimensions dimensions() const ;

  virtual LocalPoint  nodePosition( int i, int j, int k) const ;

  virtual LocalVector nodeValue( int i, int j, int k) const ;


private:

  double  thePhiMin;
  double  thePhiMax;
  MFGrid* theSectorGrid;
  double  theDelta;

  void throwUp( const char *message) const;

};

#endif
