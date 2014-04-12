#ifndef NPSTAT_PRECISETYPE_HH_
#define NPSTAT_PRECISETYPE_HH_

/*!
// \file PreciseType.h
//
// \brief Compile-time deduction of an appropriate precise numeric type
//
// Author: I. Volobouev
//
// January 2012
*/

#include <complex>

#include "Alignment/Geners/interface/IOIsNumber.hh"

namespace npstat {
    template <class T, int I=0>
    struct PreciseTypeHelper
    {
        typedef T type;
    };

    template <class T>
    struct PreciseTypeHelper<T, 1>
    {
        typedef long double type;
    };

    /**
    // Use "long double" as the most precise type among various simple
    // numeric types, std::complex<long double> for complex types, and
    // the type itself for other types.
    */
    template <class T>
    struct PreciseType
    {
        typedef typename PreciseTypeHelper<T,gs::IOIsNumber<T>::value>::type type;
    };

    template <class T>
    struct PreciseType<std::complex<T> >
    {
        typedef std::complex<long double> type;
    };

    template <class T>
    struct PreciseType<const std::complex<T> >
    {
        typedef std::complex<long double> type;
    };

    template <class T>
    struct PreciseType<volatile std::complex<T> >
    {
        typedef std::complex<long double> type;
    };

    template <class T>
    struct PreciseType<const volatile std::complex<T> >
    {
        typedef std::complex<long double> type;
    };
}

#endif // NPSTAT_PRECISETYPE_HH_

