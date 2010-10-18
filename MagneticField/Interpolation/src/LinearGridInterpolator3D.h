#ifndef Interpolation_LinearGridInterpolator3D_h
#define Interpolation_LinearGridInterpolator3D_h

/** \class LinearGridInterpolator3D
 *
 *  Linear interpolation in a regular 3D grid.
 *
 *  $Date: 2010/10/14 12:05:51 $
 *  $Revision: 1.9 $
 *  \author T. Todorov 
 */

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "MagneticField/Interpolation/src/Grid1D.h"
#include "MagneticField/Interpolation/src/Grid3D.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

#ifdef DEBUG_LinearGridInterpolator3D
#include <iostream>
#include "MagneticField/Interpolation/src/InterpolationDebug.h"
#endif

class LinearGridInterpolator3D {
public:

  typedef Grid3D::ValueType ValueType;
  typedef Grid3D::Scalar Scalar;
  typedef ValueType ReturnType; 

  LinearGridInterpolator3D( const Grid3D& g) :
    grid(g), grida(g.grida()), gridb(g.gridb()), gridc(g.gridc()) {}

  void throwGridInterpolator3DException(void);
  
  ReturnType interpolate( Scalar a, Scalar b, Scalar c); 
  //  Value operator()( Scalar a, Scalar b, Scalar c) {return interpolate(a,b,c);}

private:
  const Grid3D& grid;
  const Grid1D& grida;
  const Grid1D& gridb;
  const Grid1D& gridc;

};

#endif
