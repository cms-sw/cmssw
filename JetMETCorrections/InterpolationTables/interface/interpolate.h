#ifndef NPSTAT_INTERPOLATE_HH_
#define NPSTAT_INTERPOLATE_HH_

/*!
// \file interpolate.h
//
// \brief Low-order polynomial interpolation (up to cubic)
//        for equidistant coordinates
//
// Author: I. Volobouev
//
// October 2009
*/

#include "JetMETCorrections/InterpolationTables/interface/ProperDblFromCmpl.h"

namespace npstat {
    /**
    // Linear interpolation. Assumes that
    // the function values are given at 0 and 1.
    */
    template<typename T>
    inline T interpolate_linear(const double x, const T& f0, const T& f1)
    {
        const typename ProperDblFromCmpl<T>::type dx = 1.0 - x;
        return f0*dx + f1*static_cast<typename ProperDblFromCmpl<T>::type>(x);
    }

    /**
    // Quadratic interpolation. Assume that
    // the function values are given at 0, 1, and 2.
    */
    template<typename T>
    inline T interpolate_quadratic(const double x, const T& f0,
                                   const T& f1, const T& f2)
    {
        static const typename ProperDblFromCmpl<T>::type two = 2.0;
        const typename ProperDblFromCmpl<T>::type dx = x - 1.0;
        return f1 + ((f2 - f0)/two + ((f2 - f1) + (f0 - f1))*(dx/two))*dx;
    }

    /**
    // Cubic interpolation. Assume that
    // the function values are given at 0, 1, 2, and 3.
    */
    template<typename T>
    inline T interpolate_cubic(const double x, const T& f0, const T& f1,
                               const T& f2, const T& f3)
    {
        return interpolate_linear(x*(3.0 - x)/2.0,
                                  interpolate_linear(x/3.0, f0, f3),
                                  interpolate_linear(x - 1.0, f1, f2));
    }

    //@{
    /**
    // Get the coefficients of the interpolating polynomial.
    // The interpolated function values are provided at 0, 1, ...
    // The return value of the function is the number of
    // coefficients (i.e., the polynomial degree plus one).
    // On exit, the coefficients are placed into the "buffer"
    // array in the order of increasing monomial degree.
    // The length of the provided buffer must be sufficient
    // to hold all these coefficients.
    */
    template<typename T>
    unsigned interpolation_coefficients(T* buffer, unsigned bufLen,
                                        const T& f0, const T& f1);
    template<typename T>
    unsigned interpolation_coefficients(T* buffer, unsigned bufLen,
                                        const T& f0, const T& f1, const T& f2);
    template<typename T>
    unsigned interpolation_coefficients(T* buffer, unsigned bufLen,
                                        const T& f0, const T& f1,
                                        const T& f2, const T& f3);
    //@}
}

#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

namespace npstat {
    template<typename T>
    unsigned interpolation_coefficients(T* buffer, const unsigned bufLen,
                                        const T& f0, const T& f1)
    {
        if (bufLen <= 1U) throw npstat::NpstatInvalidArgument(
            "In npstat::interpolation_coefficients: "
            "insufficient length of the output buffer");
        buffer[0] = f0;
        buffer[1] = f1 - f0;
        return 2U;
    }

    template<typename T>
    unsigned interpolation_coefficients(T* buffer, const unsigned bufLen,
                                        const T& f0, const T& f1, const T& f2)
    {
        if (bufLen <= 2U) throw npstat::NpstatInvalidArgument(
            "In npstat::interpolation_coefficients: "
            "insufficient length of the output buffer");
        buffer[0] = f0;
        buffer[1] = static_cast<T>((f1 - f2 + 3*(f1 - f0))/2.0);
        buffer[2] = static_cast<T>(((f0 - f1) + (f2 - f1))/2.0);
        return 3U;
    }

    template<typename T>
    unsigned interpolation_coefficients(T* buffer, const unsigned bufLen,
                                        const T& f0, const T& f1,
                                        const T& f2, const T& f3)
    {
        if (bufLen <= 3U) throw npstat::NpstatInvalidArgument(
            "In npstat::interpolation_coefficients: "
            "insufficient length of the output buffer");
        buffer[0] = f0;
        buffer[1] = static_cast<T>((11*(f1 - f0)+7*(f1 - f2)+2*(f3 - f2))/6.0);
        buffer[2] = static_cast<T>((2*(f0 - f1)+3*(f2 - f1)+(f2 - f3))/2.0);
        buffer[3] = static_cast<T>(((f3 - f0) + 3*(f1 - f2))/6.0);
        return 4U;
    }
}


#endif // NPSTAT_INTERPOLATE_HH_

