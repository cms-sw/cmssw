#ifndef NPSTAT_COMPLEXCOMPARESFALSE_HH_
#define NPSTAT_COMPLEXCOMPARESFALSE_HH_

/*!
// \file ComplexComparesFalse.h
//
// \brief Ordering extended to complex numbers by always returning "false"
//
// Author: I. Volobouev
//
// January 2012
*/

#include <complex>

namespace npstat {
    /**
    // This template compares two numbers. For simple numeric types
    // (int, double, etc) the numbers themselves are compared while for
    // std::complex<...> types "false" is returned for every comparison.
    */
    template <class T>
    struct ComplexComparesFalse
    {
        inline static bool less(const T& l, const T& r)
            {return l < r;}

        inline static bool more(const T& l, const T& r)
            {return l > r;}
    };

    template <class T>
    struct ComplexComparesFalse<std::complex<T> >
    {
        inline static bool less(const std::complex<T>&, const std::complex<T>&)
            {return false;}

        inline static bool more(const std::complex<T>&, const std::complex<T>&)
            {return false;}
    };
}

#endif // NPSTAT_COMPLEXCOMPARESFALSE_HH_

