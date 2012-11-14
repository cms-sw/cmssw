#ifndef NPSTAT_ABSMULTIVARIATEFUNCTOR_HH_
#define NPSTAT_ABSMULTIVARIATEFUNCTOR_HH_

/*!
// \file AbsMultivariateFunctor.h
//
// \brief Interface definition for multidimensional functors
//
// Author: I. Volobouev
//
// May 2010
*/

namespace npstat {
    /**
    // Base class for a variety of multivariate functor-based calculations
    */
    struct AbsMultivariateFunctor
    {
        inline virtual ~AbsMultivariateFunctor() {}

        /** Function value */
        virtual double operator()(const double* point, unsigned dim) const = 0;

        /** Minimum expected dimensionality */
        virtual unsigned minDim() const = 0;

        /** 
        // Maximum expected dimensionality
        // (will typically be equal to the minimum)
        */
        virtual unsigned maxDim() const {return minDim();}
    };
}

#endif // NPSTAT_ABSMULTIVARIATEFUNCTOR_HH_

