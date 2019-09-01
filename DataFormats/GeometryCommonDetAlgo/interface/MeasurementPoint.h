#ifndef MeasurementPoint_H
#define MeasurementPoint_H

#include "DataFormats/GeometryVector/interface/MeasurementTag.h"
#include "DataFormats/GeometryVector/interface/Point2DBase.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"

typedef Point2DBase<float, MeasurementTag> Measurement2DPoint;
typedef Point3DBase<float, MeasurementTag> Measurement3DPoint;

/// Measurement points are two-dimensional by default.
typedef Measurement2DPoint MeasurementPoint;

#endif  // MeasurementPoint_H
