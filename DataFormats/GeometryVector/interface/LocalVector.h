#ifndef GeometryVector_LocalVector_h
#define GeometryVector_LocalVector_h

#include "DataFormats/GeometryVector/interface/LocalTag.h"
//#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

//typedef Vector2DBase< float, LocalTag>    Local2DVector;
typedef Vector3DBase<float, LocalTag> Local3DVector;

// Local Vectors are three-dimensional by default
typedef Local3DVector LocalVector;

#endif  // GeometryVector_LocalVector_h
