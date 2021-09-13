#ifndef NPSTAT_MULTIVARIATEFUNCTORSCANNER_HH_
#define NPSTAT_MULTIVARIATEFUNCTORSCANNER_HH_

/*!
// \file MultivariateFunctorScanner.h
//
// \brief Adapts any AbsMultivariateFunctor for use with ArrayND method
//        "functorFill"
//
// Author: I. Volobouev
//
// July 2012
*/

#include <vector>
#include <cassert>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/AbsMultivariateFunctor.h"

namespace npstat {
    /**
    // This class adapts an object derived from AbsMultivariateFunctor
    // so that it can be used with ArrayND method "functorFill" and such
    */
    template<class IndexMapper>
    class MultivariateFunctorScanner
    {
    public:
        /**
        // A mapper for each coordinate in the "maps" argument will
        // convert the array index into a proper argument for the scanned
        // density.
        //
        // This functor will NOT make copies of either "fcn" or "maps"
        // parameters. These parameters will be used by reference only
        // (aliased). It is up to the user of this class to ensure proper
        // lifetime of these objects.
        */
        inline MultivariateFunctorScanner(const AbsMultivariateFunctor& fcn,
                                          const std::vector<IndexMapper>& maps)
            : fcn_(fcn), mapping_(maps), buf_(fcn.minDim()), dim_(fcn.minDim())
        {
            if (!(dim_ && dim_ == maps.size())) throw npstat::NpstatInvalidArgument(
                "In npstat::MultivariateFunctorScanner constructor: "
                "incompatible arguments");
            if (dim_ != fcn.maxDim()) throw npstat::NpstatInvalidArgument(
                "In npstat::MultivariateFunctorScanner constructor: "
                "functors of variable dimensionality are not supported");
        }

        /** Calculate the functor value for the given array indices */
        inline double operator()(const unsigned* index,
                                 const unsigned indexLen) const
        {
            if (dim_ != indexLen) throw npstat::NpstatInvalidArgument(
                "In npstat::MultivariateFunctorScanner::operator(): "
                "incompatible input point dimensionality");
            assert(index);
            double* x = &buf_[0];
            for (unsigned i=0; i<dim_; ++i)
                x[i] = mapping_[i](index[i]);
            return fcn_(x, dim_);
        }

    private:
        MultivariateFunctorScanner();

        const AbsMultivariateFunctor& fcn_;
        const std::vector<IndexMapper>& mapping_;
        mutable std::vector<double> buf_;
        unsigned dim_;
    };
}

#endif // NPSTAT_MULTIVARIATEFUNCTORSCANNER_HH_

