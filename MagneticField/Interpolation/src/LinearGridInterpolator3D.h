#ifndef Interpolation_LinearGridInterpolator3D_h
#define Interpolation_LinearGridInterpolator3D_h

/** \class LinearGridInterpolator3D
 *
 *  Linear interpolation in a regular 3D grid.
 *
 *  $Date: $
 *  $Revision: $
 *  \author T. Todorov 
 */

#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

#ifdef DEBUG_LinearGridInterpolator3D
#include <iostream>
using namespace std;
#include "MagneticField/Interpolation/src/InterpolationDebug.h"
#endif

template <class Value, class T>
class LinearGridInterpolator3D {
public:

  typedef T  Scalar;

  LinearGridInterpolator3D( const Grid3D<Value,T>& g) :
    grid(g), grida(g.grida()), gridb(g.gridb()), gridc(g.gridc()) {}

  Value interpolate( Scalar a, Scalar b, Scalar c) {
    int i = grida.index(a);
    int j = gridb.index(b);
    int k = gridc.index(c);

    if (i==-1 || j==-1 || k==-1) {
      // point outside of grid validity!
      throw GridInterpolator3DException( grida.lower(),gridb.lower(),gridc.lower(),
					 grida.upper(),gridb.upper(),gridc.upper());
    }

    Scalar s = (a - grida.node(i)) / grida.step();
    Scalar t = (b - gridb.node(j)) / gridb.step();
    Scalar u = (c - gridc.node(k)) / gridc.step();

#ifdef DEBUG_LinearGridInterpolator3D
    if (InterpolationDebug::debug) {
      cout << "LinearGridInterpolator3D called with a,b,c " << a << "," << b << "," << c << endl;
      cout <<" i,j,k = " << i << "," << j << "," << k 
	   << " s,t,u = " << s << "," << t << "," << u << endl;
      cout << "Node positions for " << i << "," << j << "," << k << " : "
	   << grida.node(i) << "," << gridb.node(j) << "," << gridc.node(k) << endl;
      cout << "Node positions for " << i+1 << "," << j+1 << "," << k+1 << " : "
	   << grida.node(i+1) << "," << gridb.node(j+1) << "," << gridc.node(k+1) << endl;
      cout << "Grid(" << i << "," << j << "," << k << ") = " << grid(i,  j,  k) << " ";
      cout << "Grid(" << i << "," << j << "," << k+1 << ") = " << grid(i,j,k+1) << endl;
      cout << "Grid(" << i << "," << j+1 << "," << k << ") = " << grid(i,j+1,k) << " ";
      cout << "Grid(" << i << "," << j+1 << "," << k+1 << ") = " << grid(i,j+1,k+1) << endl;
      cout << "Grid(" << i+1 << "," << j << "," << k << ") = " << grid(i+1,j,k) << " ";
      cout << "Grid(" << i+1 << "," << j << "," << k+1 << ") = " << grid(i+1,j,k+1) << endl;
      cout << "Grid(" << i+1 << "," << j+1 << "," << k << ") = " << grid(i+1,j+1,k) << " ";
      cout << "Grid(" << i+1 << "," << j+1 << "," << k+1 << ") = " << grid(i+1,j+1,k+1) << endl;
      cout << "number of nodes: " << grida.nodes() << "," << gridb.nodes() << "," << gridc.nodes() << endl;

//     cout << (1-s)*(1-t)*(1-u)*grid(i,  j,  k) << " " << (1-s)*(1-t)*u*grid(i,  j,  k+1) << endl 
// 	 << (1-s)*   t *(1-u)*grid(i,  j+1,k) << " " << (1-s)*   t *u*grid(i,  j+1,k+1) << endl 
// 	 << s    *(1-t)*(1-u)*grid(i+1,j,  k) << " " <<  s    *(1-t)*u*grid(i+1,j,  k+1) << endl 
// 	 << s    *   t *(1-u)*grid(i+1,j+1,k) << " " << s    *   t *u*grid(i+1,j+1,k+1) << endl;
    }

#endif

    Value result = 
      (1-s)*(1-t)*(1-u)*grid(i,  j,  k) + (1-s)*(1-t)*u*grid(i,  j,  k+1) + 
      (1-s)*   t *(1-u)*grid(i,  j+1,k) + (1-s)*   t *u*grid(i,  j+1,k+1) +
      s    *(1-t)*(1-u)*grid(i+1,j,  k) + s    *(1-t)*u*grid(i+1,j,  k+1) + 
      s    *   t *(1-u)*grid(i+1,j+1,k) + s    *   t *u*grid(i+1,j+1,k+1);

    return result;
  }

  //  Value operator()( Scalar a, Scalar b, Scalar c) {return interpolate(a,b,c);}

private:
  const Grid3D<Value,T>& grid;
  const Grid1D<T>& grida;
  const Grid1D<T>& gridb;
  const Grid1D<T>& gridc;

};

#endif
