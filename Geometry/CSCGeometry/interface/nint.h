#ifndef GEOMETRY_CSCGeometry_NINT_H_
#define GEOMETRY_CSCGeometry_NINT_H_

// From CommonDet/Utilities

/// Return the nearest integer - analogous to the FORTRAN intrinsic NINT

inline int nint( float a)  { return a>=0 ? int( a+0.5) : int( a-0.5);}
inline int nint( double a) { return a>=0 ? int( a+0.5) : int( a-0.5);}

#endif
