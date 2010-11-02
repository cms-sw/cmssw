#include "MagneticField/Interpolation/src/LinearGridInterpolator3D.h"
#include "MagneticField/Interpolation/src/Grid1D.h"
#include "MagneticField/Interpolation/src/Grid3D.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

void
LinearGridInterpolator3D::throwGridInterpolator3DException(void)
{
  throw GridInterpolator3DException(grida.lower(),gridb.lower(),gridc.lower(),
                                    grida.upper(),gridb.upper(),gridc.upper());
}

LinearGridInterpolator3D::ValueType 
LinearGridInterpolator3D::interpolate( Scalar a, Scalar b, Scalar c) 
{
  int i = grida.index(a);
  int j = gridb.index(b);
  int k = gridc.index(c);

  if (i==-1 || j==-1 || k==-1) {
    // point outside of grid validity!
    throwGridInterpolator3DException();
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
//       << (1-s)*   t *(1-u)*grid(i,  j+1,k) << " " << (1-s)*   t *u*grid(i,  j+1,k+1) << endl 
//       << s    *(1-t)*(1-u)*grid(i+1,j,  k) << " " <<  s    *(1-t)*u*grid(i+1,j,  k+1) << endl 
//       << s    *   t *(1-u)*grid(i+1,j+1,k) << " " << s    *   t *u*grid(i+1,j+1,k+1) << endl;
  }

#endif

  //chances are this is more numerically precise this way
  ValueType result = (1-s)*(1-t)*u*(grid(i,  j,  k+1) - grid(i,  j,  k));
  result +=      (1-s)*   t *u*(grid(i,  j+1,k+1) - grid(i,  j+1,k));
  result +=      s    *(1-t)*u*(grid(i+1,j,  k+1) - grid(i+1,j,  k));
  result +=      s    *   t *u*(grid(i+1,j+1,k+1) - grid(i+1,j+1,k)); 
  result += (1-s)*t*(grid(i,  j+1,k)-grid(i,  j,  k));
  result += s    *t*(grid(i+1,j+1,k)-grid(i+1,j,  k));
  result += s*(grid(i+1,j,  k)-grid(i,  j,  k));
  result += grid(i,  j,  k);
  //      (1-s)*(1-t)*(1-u)*grid(i,  j,  k) + (1-s)*(1-t)*u*grid(i,  j,  k+1) + 
  //      (1-s)*   t *(1-u)*grid(i,  j+1,k) + (1-s)*   t *u*grid(i,  j+1,k+1) +
  //      s    *(1-t)*(1-u)*grid(i+1,j,  k) + s    *(1-t)*u*grid(i+1,j,  k+1) + 
  //      s    *   t *(1-u)*grid(i+1,j+1,k) + s    *   t *u*grid(i+1,j+1,k+1);

  return result;
}
