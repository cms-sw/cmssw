#ifndef MeasurementVector_H
#define MeasurementVector_H

#include "DataFormats/GeometryVector/interface/MeasurementTag.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

typedef Vector2DBase< float, MeasurementTag>    Measurement2DVector;
typedef Vector3DBase< float, MeasurementTag>    Measurement3DVector;

/// Measurement Vectors are three-dimensional by default.
typedef Measurement3DVector                     MeasurementVector;

#endif // MeasurementVector_H
