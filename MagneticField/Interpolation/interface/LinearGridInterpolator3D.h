// #define DEBUG_LinearGridInterpolator3D

#ifndef LinearGridInterpolator3D_H
#define LinearGridInterpolator3D_H

#ifdef DEBUG_LinearGridInterpolator3D
#include <iostream>
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

    Scalar s = (a - grida.node(i)) / grida.step();
    Scalar t = (b - gridb.node(j)) / gridb.step();
    Scalar u = (c - gridc.node(k)) / gridc.step();

#ifdef DEBUG_LinearGridInterpolator3D
    cout << "LinearGridInterpolator3D called with a,b,c " << a << "," << b << "," << c << endl;
    cout <<" i,j,k = " << i << "," << j << "," << k 
	 << " s,t,u = " << s << "," << t << "," << u << endl;
    cout << "Node positions for " << i << "," << j << "," << k << " : "
	 << grida.node(i) << "," << gridb.node(j) << "," << gridc.node(k) << endl;
    cout << "Node positions for " << i+1 << "," << j+1 << "," << k+1 << " : "
	 << grida.node(i+1) << "," << gridb.node(j+1) << "," << gridc.node(k+1) << endl;
    cout << "Grid(" << i << "," << j << "," << k << ") = " << grid(i,  j,  k) << endl;
    cout << "Grid(" << i+1 << "," << j+1 << "," << k+1 << ") = " << grid(i+1,j+1,k+1) << endl;
    cout << "Grid(" << i << "," << j+1 << "," << k << ") = " << grid(i,j+1,k) << endl;
    cout << "number of nodes: " << grida.nodes() << "," << gridb.nodes() << "," << gridc.nodes() << endl;
#endif

    Value result = 
      (1-s)*(1-t)*(1-u)*grid(i,  j,  k) + (1-s)*(1-t)*u*grid(i,  j,  k+1) + 
      (1-s)*   t *(1-u)*grid(i,  j+1,k) + (1-s)*   t *u*grid(i,  j+1,k+1) +
      s    *(1-t)*(1-u)*grid(i+1,j,  k) + s    *(1-t)*u*grid(i+1,j,  k+1) + 
      s    *   t *(1-u)*grid(i+1,j+1,k) + s    *   t *u*grid(i+1,j+1,k+1);
    return result;
  }

  Value operator()( Scalar a, Scalar b, Scalar c) {return interpolate(a,b,c);}

private:

  const Grid3D<Value,T>& grid;
  const Grid1D<T>& grida;
  const Grid1D<T>& gridb;
  const Grid1D<T>& gridc;

};

#endif
