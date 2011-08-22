#include "../interface/AsymPow.h"

#include <cmath>
#include <cassert>
#include <cstdio>

AsymPow::AsymPow(const char *name, const char *title, RooAbsReal &kappaLow, RooAbsReal &kappaHigh, RooAbsReal &theta) :
        RooAbsReal(name,title),
        kappaLow_("kappaLow","Base for theta < 0", this, kappaLow), 
        kappaHigh_("kappaHigh","Base for theta > 0", this, kappaHigh),
        theta_("theta", "Exponent (unit gaussian)", this, theta) 
        { }

AsymPow::~AsymPow() {}

TObject *AsymPow::clone(const char *newname) const 
{
    // never understood if RooFit actually cares of const-correctness or not.
    return new AsymPow(newname, this->GetTitle(), 
                const_cast<RooAbsReal &>(kappaLow_.arg()), 
                const_cast<RooAbsReal &>(kappaHigh_.arg()),
                const_cast<RooAbsReal &>(theta_.arg()));
}

Double_t AsymPow::evaluate() const {
    Double_t x = theta_;
    return exp(logKappaForX(x) * x);
}

Double_t AsymPow::logKappaForX(Double_t x) const {
#if 0
    // old version with discontinuous derivatives
    return (x >= 0 ? log(kappaHigh_) : - log(kappaLow_));
#else
    if (fabs(x) >= 0.5) return (x >= 0 ? log(kappaHigh_) : - log(kappaLow_));
    // interpolate between log(kappaHigh) and -log(kappaLow) 
    //    logKappa(x) = avg + halfdiff * h(2x)
    // where h(x) is the 3th order polynomial
    //    h(x) = (3 x^5 - 10 x^3 + 15 x)/8;
    // chosen so that h(x) satisfies the following:
    //      h (+/-1) = +/-1 
    //      h'(+/-1) = 0
    //      h"(+/-1) = 0
    double logKhi =  log(kappaHigh_);
    double logKlo = -log(kappaLow_);
    double avg = 0.5*(logKhi + logKlo), halfdiff = 0.5*(logKhi - logKlo);
    double twox = x+x, twox2 = twox*twox;
    double alpha = 0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.);
    double ret = avg + alpha*halfdiff;
    //assert(alpha >= -1 && alpha <= 1 && "Something is wrong in the interpolation");
    return ret;
#endif
} 

ClassImp(AsymPow)
