#ifndef NPSTAT_COMPLEXCOMPARESABS_HH_
#define NPSTAT_COMPLEXCOMPARESABS_HH_

/*!
// \file ComplexComparesAbs.h
//
// \brief Ordering extended to complex numbers by comparing their magnitudes
//
// Author: I. Volobouev
//
// January 2012
*/

#include <cmath>
#include <complex>

namespace npstat {
    /**
    // This template compares two numbers. For simple numeric types
    // (int, double, etc) the numbers themselves are compared while
    // for std::complex<...> types absolute values are compared.
    */
    template <class T>
    struct ComplexComparesAbs
    {
        inline static bool less(const T& l, const T& r)
            {return l < r;}

        inline static bool more(const T& l, const T& r)
            {return l > r;}
    };

    template <class T>
    struct ComplexComparesAbs<std::complex<T> >
    {
        inline static bool less(const std::complex<T>& l,
                                const std::complex<T>& r)
            {return std::abs(l) < std::abs(r);}

        inline static bool more(const std::complex<T>& l,
                                const std::complex<T>& r)
            {return std::abs(l) > std::abs(l);}
    };
}

#endif // NPSTAT_COMPLEXCOMPARESABS_HH_

