#ifndef NPSTAT_EQUIDISTANTSEQUENCE_HH_
#define NPSTAT_EQUIDISTANTSEQUENCE_HH_

/*!
// \file EquidistantSequence.h
//
// \brief Equidistant sequences of points in either linear or log space
//
// Author: I. Volobouev
//
// March 2009
*/

#include <vector>

namespace npstat {
    /**
    // A sequence of points equidistant in linear space. Note that
    // std::vector destructor is not virtual, so do not destroy this
    // class by base pointer or reference.
    */
    class EquidistantInLinearSpace : public std::vector<double>
    {
    public:
        EquidistantInLinearSpace(double minScale, double maxScale,
                                 unsigned nScales);
        virtual ~EquidistantInLinearSpace() {}

    private:
        EquidistantInLinearSpace();
    };

    /**
    // A sequence of points equidistant in log space. Note that
    // std::vector destructor is not virtual, so do not destroy this
    // class by base pointer or reference.
    */
    class EquidistantInLogSpace : public std::vector<double>
    {
    public:
        EquidistantInLogSpace(double minScale, double maxScale,
                              unsigned nScales);
        virtual ~EquidistantInLogSpace() {}

    private:
        EquidistantInLogSpace();
    };
}

#endif // NPSTAT_EQUIDISTANTSEQUENCE_HH_

