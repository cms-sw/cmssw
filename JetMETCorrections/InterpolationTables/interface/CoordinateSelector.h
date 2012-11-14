#ifndef NPSTAT_COORDINATESELECTOR_HH_
#define NPSTAT_COORDINATESELECTOR_HH_

/*!
// \file CoordinateSelector.h
//
// \brief Multidimensional functor which picks one of the elements
//        from an array of doubles
//
// Author: I. Volobouev
//
// August 2012
*/

#include <climits>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/AbsMultivariateFunctor.h"

namespace npstat {
    /**
    // A trivial implementation of AbsMultivariateFunctor which selects
    // an element with a certain index from the input array
    */
    class CoordinateSelector : public AbsMultivariateFunctor
    {
    public:
        inline explicit CoordinateSelector(const unsigned i) : index_(i) {}

        inline double operator()(const double* point, const unsigned dim) const
        {
            if (dim <= index_)
                throw npstat::NpstatInvalidArgument(
                    "In npstat::CoordinateSelector::operator(): "
                    "input array dimensionality is too small");
            return point[index_];
        }
        inline unsigned minDim() const {return index_ + 1U;}
        inline unsigned maxDim() const {return UINT_MAX;}

    private:
        CoordinateSelector();
        unsigned index_;
    };
}

#endif // NPSTAT_COORDINATESELECTOR_HH_

