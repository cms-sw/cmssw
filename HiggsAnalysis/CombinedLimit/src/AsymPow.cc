#include "HiggsAnalysis/CombinedLimit/interface/AsymPow.h"

#include <cmath>

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
    return x >= 0 ? pow(kappaHigh_, x) : pow(kappaLow_, -x);
}

ClassImp(AsymPow)
