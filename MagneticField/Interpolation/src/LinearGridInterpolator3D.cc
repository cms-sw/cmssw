#include "LinearGridInterpolator3D.h"
#include "Grid1D.h"
#include "Grid3D.h"
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
  /*
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

  */

  Scalar s, t, u;
  int i = grida.index(a,s);
  int j = gridb.index(b,t);
  int k = gridc.index(c,u);
  
  // test range??

  grida.normalize(i,s);
  gridb.normalize(j,t);
  gridc.normalize(k,u);

#ifdef DEBUG_LinearGridInterpolator3D
  if (InterpolationDebug::debug) {
    using std::cout;
    using std::endl;
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

  int ind = grid.index(i,j,k);
  int s1 = grid.stride1(); 
  int s2 = grid.stride2(); 
  int s3 = grid.stride3(); 
  //chances are this is more numerically precise this way



  // this code for test to check  properly inline of wrapped math...
#if defined(CMS_TEST_RAWSSE)

  __m128 resultSIMD =                 _mm_mul_ps(_mm_set1_ps((1.f - s) * (1.f - t) * u), _mm_sub_ps(grid(ind           + s3).v.vec, grid(ind          ).v.vec));
  resultSIMD = _mm_add_ps(resultSIMD, _mm_mul_ps(_mm_set1_ps((1.f - s) *         t * u), _mm_sub_ps(grid(ind      + s2 + s3).v.vec, grid(ind      + s2).v.vec)));
  resultSIMD = _mm_add_ps(resultSIMD, _mm_mul_ps(_mm_set1_ps(        s * (1.f - t) * u), _mm_sub_ps(grid(ind + s1      + s3).v.vec, grid(ind + s1     ).v.vec)));
  resultSIMD = _mm_add_ps(resultSIMD, _mm_mul_ps(_mm_set1_ps(        s *         t * u), _mm_sub_ps(grid(ind + s1 + s2 + s3).v.vec, grid(ind + s1 + s2).v.vec)));
  resultSIMD = _mm_add_ps(resultSIMD, _mm_mul_ps(_mm_set1_ps((1.f - s) *         t    ), _mm_sub_ps(grid(ind      + s2     ).v.vec, grid(ind          ).v.vec)));
  resultSIMD = _mm_add_ps(resultSIMD, _mm_mul_ps(_mm_set1_ps(        s *         t    ), _mm_sub_ps(grid(ind + s1 + s2     ).v.vec, grid(ind + s1     ).v.vec)));
  resultSIMD = _mm_add_ps(resultSIMD, _mm_mul_ps(_mm_set1_ps(        s                ), _mm_sub_ps(grid(ind + s1          ).v.vec, grid(ind          ).v.vec)));
  resultSIMD = _mm_add_ps(resultSIMD,                                                                                             grid(ind          ).v.vec);
  
  ValueType result; result.v=resultSIMD;


#else
  
  ValueType result = ((1.f-s)*(1.f-t)*u)*(grid(ind      +s3) - grid(ind      ));
  result =  result + ((1.f-s)*     t *u)*(grid(ind   +s2+s3) - grid(ind   +s2));
  result =  result + (s      *(1.f-t)*u)*(grid(ind+s1   +s3) - grid(ind+s1   ));
  result =  result + (s      *     t *u)*(grid(ind+s1+s2+s3) - grid(ind+s1+s2)); 
  result =  result + (        (1.f-s)*t)*(grid(ind   +s2   ) - grid(ind      ));
  result =  result + (      s        *t)*(grid(ind+s1+s2   ) - grid(ind+s1   ));
  result =  result + (                s)*(grid(ind+s1      ) - grid(ind      ));
  result =  result +                                           grid(ind      );


#endif



  //      (1-s)*(1-t)*(1-u)*grid(i,  j,  k) + (1-s)*(1-t)*u*grid(i,  j,  k+1) + 
  //      (1-s)*   t *(1-u)*grid(i,  j+1,k) + (1-s)*   t *u*grid(i,  j+1,k+1) +
  //      s    *(1-t)*(1-u)*grid(i+1,j,  k) + s    *(1-t)*u*grid(i+1,j,  k+1) + 
  //      s    *   t *(1-u)*grid(i+1,j+1,k) + s    *   t *u*grid(i+1,j+1,k+1);


  return result;


}
