#ifndef NPSTAT_ABSDIFFERENCE_HH_
#define NPSTAT_ABSDIFFERENCE_HH_

/*!
// \file absDifference.h
//
// \brief Calculate absolute value of a difference between two numbers
//        for an extended set of types
//
// Author: I. Volobouev
//
// October 2009
*/

#include <cmath>
#include <complex>

#include "Alignment/Geners/interface/IOIsUnsigned.hh"

namespace npstat {
    namespace Private {
        template <typename T>
        struct AbsReturnType
        {
            typedef T type;
        };

        template <typename T>
        struct AbsReturnType<std::complex<T> >
        {
            typedef T type;
        };

        template <typename T>
        struct AbsReturnType<const std::complex<T> >
        {
            typedef T type;
        };

        template <typename T>
        struct AbsReturnType<volatile std::complex<T> >
        {
            typedef T type;
        };

        template <typename T>
        struct AbsReturnType<const volatile std::complex<T> >
        {
            typedef T type;
        };

        // Signed type
        template <typename T, int Unsigned=0>
        struct AbsHelper
        {
            typedef typename Private::AbsReturnType<T>::type return_type;

            inline static return_type delta(const T& v1, const T& v2)
                {return std::abs(v1 - v2);}

            inline static return_type value(const T& v1)
                {return std::abs(v1);}
        };

        // Unsigned type
        template <typename T>
        struct AbsHelper<T, 1>
        {
            typedef typename Private::AbsReturnType<T>::type return_type;

            inline static return_type delta(const T& v1, const T& v2)
                {return v1 > v2 ? v1 - v2 : v2 - v1;}

            inline static return_type value(const T& v1)
                {return v1;}
        };
    }

    /**
    // Absolute value of the difference between two numbers.
    // Works for all standard numeric types, including unsigned and complex.
    */
    template<typename T>
    inline typename Private::AbsReturnType<T>::type
    absDifference(const T& v1, const T& v2)
    {
        return Private::AbsHelper<T,gs::IOIsUnsigned<T>::value>::delta(v1, v2);
    }

    /**
    // Absolute value of a number.
    // Works for all standard numeric types, including unsigned and complex.
    */
    template<typename T>
    inline typename Private::AbsReturnType<T>::type
    absValue(const T& v1)
    {
        return Private::AbsHelper<T,gs::IOIsUnsigned<T>::value>::value(v1);
    }
}

#endif // NPSTAT_ABSDIFFERENCE_HH_

