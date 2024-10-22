#ifndef GeometryVector_GlobalVector_h
#define GeometryVector_GlobalVector_h

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

typedef Vector3DBase<float, GlobalTag> Global3DVector;

// Global Vectors are three-dimensional by default
typedef Global3DVector GlobalVector;

#endif  // GeometryVector_GlobalVector_h
