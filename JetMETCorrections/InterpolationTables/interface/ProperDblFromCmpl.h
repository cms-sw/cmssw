#ifndef NPSTAT_PROPERDBLFROMCMPL_HH_
#define NPSTAT_PROPERDBLFROMCMPL_HH_

/*!
// \file ProperDblFromCmpl.h
//
// \brief Compile-time deduction of the underlying floating point type from
//        the given complex type
//
// Author: I. Volobouev
//
// January 2012
*/

#include <complex>

namespace npstat {
    template <class T>
    struct ProperDblFromCmpl
    {
        typedef double type;
    };

    template <class T>
    struct ProperDblFromCmpl<std::complex<T> >
    {
        typedef T type;
    };

    template <class T>
    struct ProperDblFromCmpl<const std::complex<T> >
    {
        typedef T type;
    };

    template <class T>
    struct ProperDblFromCmpl<volatile std::complex<T> >
    {
        typedef T type;
    };

    template <class T>
    struct ProperDblFromCmpl<const volatile std::complex<T> >
    {
        typedef T type;
    };
}

#endif // NPSTAT_PROPERDBLFROMCMPL_HH_

