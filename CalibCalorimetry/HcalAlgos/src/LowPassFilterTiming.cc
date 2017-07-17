#include <cmath>
#include <cassert>
#include "FWCore/Utilities/interface/Exception.h"

#include "CalibCalorimetry/HcalAlgos/interface/LowPassFilterTiming.h"

#define NPARAMETERS 6U

//
// The formula below join_location - h is aleft*(x_in - join_location) + bleft.
// The formula above join_location + h is aright*(x_in - join_location) + bright.
// Smooth cubic interpolation is performed within +-h of join_location.
//
static double two_joined_lines(const double x_in, const double aleft,
                               const double bleft, const double aright,
                               const double bright, const double join_location,
                               const double h)
{
    assert(h >= 0.0);
    const double x = x_in - join_location;
    if (x <= -h)
        return aleft*x + bleft;
    else if (x >= h)
        return aright*x + bright;
    else
    {
        const double vleft = -h*aleft + bleft;
        const double vright = aright*h + bright;
        const double b = (aright - aleft)/4.0/h;
        const double d = (vright + vleft - 2*b*h*h)/2.0;
        const double a = (d + aright*h - b*h*h - vright)/(2.0*h*h*h);
        const double c = -(3*d + aright*h + b*h*h - 3*vright)/(2.0*h);
        return ((a*x + b)*x + c)*x + d;
    }
}

unsigned LowPassFilterTiming::nParameters() const
{
    return NPARAMETERS;
}

//
// The time constant decreases linearly in the space of log(V + Vbias),
// from tauMin+tauDelta when V = 0 to tauMin when V = V1.
// Around log(V1 + Vbias), the curve is joined by a third order polynomial.
// The width of the join is dLogV on both sides of log(V1 + Vbias).
//
// Value "tauDelta" = 0 can be used to create a constant time filter.
//
double LowPassFilterTiming::operator()(const double v,
                                       const double* pars,
                                       const unsigned nParams) const
{
    assert(nParams == NPARAMETERS);
    assert(pars);
    unsigned ipar = 0;
    const double logVbias = pars[ipar++];
    const double logTauMin = pars[ipar++];

    // The middle of the join region. Not a log actually,
    // it is simple in the log space.
    const double logV0     = pars[ipar++];

    // Log of the width of the join region
    const double logDelta  = pars[ipar++];

    // Log of the minus negative slope
    const double slopeLog  = pars[ipar++];

    // Log of the maximum delay time (cutoff)
    const double tauMax    = pars[ipar++];
    assert(ipar == NPARAMETERS);

    // What happens for large (in magnitude) negative voltage inputs?
    const double Vbias = exp(logVbias);
    const double shiftedV = v + Vbias;
    if (shiftedV <= 0.0)
        return tauMax;

    const double lg = log(shiftedV);
    const double delta = exp(logDelta);
    const double tauMin = exp(logTauMin);
    double result = two_joined_lines(lg, -exp(slopeLog), tauMin,
                                     0.0, tauMin, logV0, delta);
    if (result > tauMax)
        result = tauMax;
    return result;
}
