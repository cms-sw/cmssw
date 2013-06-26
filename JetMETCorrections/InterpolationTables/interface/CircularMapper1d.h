#ifndef NPSTAT_CIRCULARMAPPER1D_HH_
#define NPSTAT_CIRCULARMAPPER1D_HH_

/*!
// \file CircularMapper1d.h
//
// \brief Linear transformation for circular topology
//
// Author: I. Volobouev
//
// June 2012
*/

#include <cmath>

#include "JetMETCorrections/InterpolationTables/interface/LinearMapper1d.h"

namespace npstat {
    /**
    // 1-d linear transformation functor followed by the shift of
    // the result into the interval [-T/2, T/2], where T is the period
    */
    class CircularMapper1d
    {
    public:
        inline CircularMapper1d() : a_(1.0), b_(0.0), period_(2.0*M_PI) {}

        inline CircularMapper1d(const double ca, const double cb,
                                const double cperiod)
            : a_(ca), b_(cb), period_(std::abs(cperiod)) {check();}

        inline CircularMapper1d(const LinearMapper1d& mapper,
                                const double cperiod)
            : a_(mapper.a()), b_(mapper.b()),
              period_(std::abs(cperiod)) {check();}

        inline double operator()(const double& x) const
        {
            double value = a_*x + b_;
            value -= period_*floor(value/period_);
            if (value > period_/2.0)
                value -= period_;
            return value;
        }

        inline double a() const {return a_;}
        inline double b() const {return b_;}
        inline double period() const {return period_;}
        inline LinearMapper1d linearMapper() const
            {return LinearMapper1d(a_, b_);}

    private:
        inline void check()
        {
            if (!period_) throw npstat::NpstatInvalidArgument(
                "In npstat::CircularMapper1d constructor: "
                "invalid period argument (can not be 0)");
        }

        double a_;
        double b_;
        double period_;
    };
}

#endif // NPSTAT_CIRCULARMAPPER1D_HH_

