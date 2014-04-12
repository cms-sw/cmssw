#ifndef NPSTAT_ABSARRAYPROJECTOR_HH_
#define NPSTAT_ABSARRAYPROJECTOR_HH_

/*!
// \file AbsArrayProjector.h
//
// \brief Interface definition for functors used to make array projections
//
// Author: I. Volobouev
//
// March 2010
*/

namespace npstat {
    /**
    // Interface class for piecemeal processing of array data and coordinates.
    // Intended for making multidimensional array projections.
    */
    template <typename Input, typename Result>
    struct AbsArrayProjector
    {
        virtual ~AbsArrayProjector() {}

        /** Clear all accumulated results */
        virtual void clear() = 0;

        /** Process one array point */
        virtual void process(const unsigned *index, unsigned indexLen,
                             unsigned long linearIndex,
                             const Input& value) = 0;

        /** Return the result at the end of array processing */
        virtual Result result() = 0;
    };
}

#endif // ABSARRAYPROJECTOR_HH_

