#ifndef NPSTAT_INTERPOLATIONFUNCTORINSTANCES_HH_
#define NPSTAT_INTERPOLATIONFUNCTORINSTANCES_HH_

/*!
// \file InterpolationFunctorInstances.h
//
// \brief Typedefs for some common uses of the StorableInterpolationFunctor
//        template
//
// Author: I. Volobouev
//
// September 2012
*/

#include "JetMETCorrections/InterpolationTables/interface/StorableInterpolationFunctor.h"
#include "JetMETCorrections/InterpolationTables/interface/DualAxis.h"

namespace npstat {
    typedef StorableInterpolationFunctor<double,DualAxis>
        DoubleInterpolationFunctor;

    typedef StorableInterpolationFunctor<double,UniformAxis>
        DoubleUAInterpolationFunctor;

    typedef StorableInterpolationFunctor<double,GridAxis>
        DoubleNUInterpolationFunctor;

    typedef StorableInterpolationFunctor<float,DualAxis>
        FloatInterpolationFunctor;

    typedef StorableInterpolationFunctor<float,UniformAxis>
        FloatUAInterpolationFunctor;

    typedef StorableInterpolationFunctor<float,GridAxis>
        FloatNUInterpolationFunctor;
}

#endif // NPSTAT_INTERPOLATIONFUNCTORINSTANCES_HH_

