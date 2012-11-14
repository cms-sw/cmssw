#ifndef NPSTAT_ARRAYRANGE_HH_
#define NPSTAT_ARRAYRANGE_HH_

/*!
// \file ArrayRange.h
//
// \brief Multidimensional range of array indices
//
// Author: I. Volobouev
//
// October 2009
*/

#include "JetMETCorrections/InterpolationTables/interface/ArrayShape.h"
#include "JetMETCorrections/InterpolationTables/interface/BoxND.h"

namespace npstat {
    /**
    // Utility class for use in certain array iterations
    */
    struct ArrayRange : public BoxND<unsigned>
    {
        inline ArrayRange() {}

        /** Constructor from a given number of dimensions */
        inline ArrayRange(unsigned dim) : BoxND<unsigned>(dim) {}

        /** The given interval is repeated for every dimension */
        inline ArrayRange(unsigned dim, const Interval<unsigned>& r1)
            : BoxND<unsigned>(dim, r1) {}

        //@{
        /**
        // Constructor which creates a range out of a shape
        // which is used to represent the upper limit. The
        // lower limit in each dimension is set to 0.
        */
        inline ArrayRange(const ArrayShape& shape) : BoxND<unsigned>(shape) {}
        ArrayRange(const unsigned* shape, unsigned shapeLen);
        //@}

        /**
        // The shape which corresponds to this range
        // (i.e., max - min in all dimensions)
        */
        ArrayShape shape() const;

        //@{
        /** Check for compatibility with a shape */
        bool isCompatible(const ArrayShape& shape) const;
        bool isCompatible(const unsigned* shape, unsigned shapeLen) const;
        //@}

        /** How many elements will be iterated over? */
        unsigned long rangeSize() const;

        /** Operator for use with maps */
        bool operator<(const ArrayRange&) const;

        /**
        // This method changes the range of this object so that
        // for each dimension the minimum becomes larger by 1 and the
        // maximum smaller by 1.
        */
        ArrayRange& stripOuterLayer();

        /**
        // Get the lower range limits into an array. The length of
        // the limit array should be at least equal to the dimensionality.
        */
        void lowerLimits(unsigned* limits, unsigned limitsLen) const;

        /** Get the upper range limits into an array */
        void upperLimits(unsigned* limits, unsigned limitsLen) const;

        /** Get the range into an array */
        void rangeLength(unsigned* range, unsigned rangeLen) const;
    };
}

#endif // NPSTAT_ARRAYRANGE_HH_

