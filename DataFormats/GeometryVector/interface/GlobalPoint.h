#ifndef GeometryVector_GlobalPoint_h
#define GeometryVector_GlobalPoint_h

#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"

typedef Point3DBase< float, GlobalTag>    Global3DPoint;

// Global points are three-dimensional by default
typedef Global3DPoint                     GlobalPoint;


#endif // GeometryVector_GlobalPoint_h
