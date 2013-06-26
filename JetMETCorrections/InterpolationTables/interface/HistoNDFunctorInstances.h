#ifndef NPSTAT_HISTONDFUNCTORINSTANCES_HH_
#define NPSTAT_HISTONDFUNCTORINSTANCES_HH_

/*!
// \file HistoNDFunctorInstances.h
//
// \brief Typedefs for some common uses of the StorableHistoNDFunctor template
//
// Author: I. Volobouev
//
// September 2012
*/

#include "JetMETCorrections/InterpolationTables/interface/StorableHistoNDFunctor.h"
#include "JetMETCorrections/InterpolationTables/interface/DualHistoAxis.h"

namespace npstat {
    typedef StorableHistoNDFunctor<double,DualHistoAxis> DoubleHistoNDFunctor;

    typedef StorableHistoNDFunctor<double,HistoAxis> DoubleUAHistoNDFunctor;

    typedef StorableHistoNDFunctor<double,NUHistoAxis> DoubleNUHistoNDFunctor;

    typedef StorableHistoNDFunctor<float,DualHistoAxis> FloatHistoNDFunctor;

    typedef StorableHistoNDFunctor<float,HistoAxis> FloatUAHistoNDFunctor;

    typedef StorableHistoNDFunctor<float,NUHistoAxis> FloatNUHistoNDFunctor;
}

#endif // NPSTAT_HISTONDFUNCTORINSTANCES_HH_

