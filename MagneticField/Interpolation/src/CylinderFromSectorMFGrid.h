#ifndef CylinderFromSectorMFGrid_H
#define CylinderFromSectorMFGrid_H

#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "FWCore/Utilities/interface/Visibility.h"

class dso_internal CylinderFromSectorMFGrid : public MFGrid {
public:

  CylinderFromSectorMFGrid(const GloballyPositioned<float>& vol,
			   double phiMin, double phiMax, MFGrid* sectorGrid);

  ~CylinderFromSectorMFGrid() override;

  LocalVector valueInTesla( const LocalPoint& p) const override;

  void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const override ;

  LocalPoint fromGridFrame( double a, double b, double c) const override ;

  Dimensions dimensions() const override;

  LocalPoint  nodePosition( int i, int j, int k) const override ;

  LocalVector nodeValue( int i, int j, int k) const override ;


private:

  double  thePhiMin;
  double  thePhiMax;
  MFGrid* theSectorGrid;
  double  theDelta;

  void throwUp( const char *message) const;

};

#endif
