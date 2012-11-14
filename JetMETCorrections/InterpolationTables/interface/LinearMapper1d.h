#ifndef NPSTAT_LINEARMAPPER1D_HH_
#define NPSTAT_LINEARMAPPER1D_HH_

/*!
// \file LinearMapper1d.h
//
// \brief Linear transformation functor
//
// Author: I. Volobouev
//
// October 2009
*/

#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

namespace npstat {
    /** Functor which performs linear mapping in 1-d */
    class LinearMapper1d
    {
    public:
        /** Default constructor builds an identity transformation */
        inline LinearMapper1d() : a_(1.0), b_(0.0) {}

        /**
        // Transform definition from two points. The point at x0
        // is mapped into y0, the point at x1 is mapped into y1.
        // The linear transformation is thus fully defined.
        */
        inline LinearMapper1d(const double x0, const double y0,
                              const double x1, const double y1)
        {
            const double dx = x1 - x0;
            if (!dx) throw npstat::NpstatInvalidArgument(
                "In npstat::LinearMapper1d constructor: "
                "invalid arguments (x0 == x1)");
            a_ = (y1 - y0)/dx;
            b_ = ((y0 + y1) - a_*(x0 + x1))/2.0;
        }

        /** Explicitly provide the transform coefficients as in y = ca*x + cb */
        inline LinearMapper1d(const double ca, const double cb)
            : a_(ca), b_(cb) {}

        /** Perform the transformation */
        inline double operator()(const double& x) const {return a_*x + b_;}

        /** Get the linear coefficient of the transform */
        inline double a() const {return a_;}

        /** Get the transform constant */
        inline double b() const {return b_;}

        /** Create the inverse transform */
        inline LinearMapper1d inverse() const
        {
            if (!a_) throw npstat::NpstatInvalidArgument(
                "In npstat::LinearMapper1d::inverse: "
                "mapping is not invertible");
            return LinearMapper1d(1.0/a_, -b_/a_);
        }

        /** Sequence of two transforms: the one on the right is applied first */
        inline LinearMapper1d operator*(const LinearMapper1d& r) const
        {
            return LinearMapper1d(a_*r.a_, a_*r.b_ + b_);
        }

    private:
        double a_;
        double b_;
    };
}

#endif // NPSTAT_LINEARMAPPER1D_HH_

