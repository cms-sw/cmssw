#ifndef GeometryVector_LocalPoint_h
#define GeometryVector_LocalPoint_h

#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/Point2DBase.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"

typedef Point2DBase< float, LocalTag>    Local2DPoint;
typedef Point3DBase< float, LocalTag>    Local3DPoint;

typedef Local3DPoint                     LocalPoint;

#endif // GeometryVector_LocalPoint_h
