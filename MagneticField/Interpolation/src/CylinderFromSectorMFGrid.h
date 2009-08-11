#ifndef CylinderFromSectorMFGrid_H
#define CylinderFromSectorMFGrid_H

#include "MagneticField/Interpolation/interface/MFGrid.h"

class CylinderFromSectorMFGrid : public MFGrid {
public:

  CylinderFromSectorMFGrid(const GloballyPositioned<float>& vol,
			   double phiMin, double phiMax, MFGrid* sectorGrid);

  ~CylinderFromSectorMFGrid();

  virtual LocalVector valueInTesla( const LocalPoint& p) const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const ;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const ;

  virtual std::vector<int> dimensions() const ;

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
