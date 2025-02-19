#include "../interface/RooSpline1D.h"

#include <stdexcept>

RooSpline1D::RooSpline1D(const char *name, const char *title, RooAbsReal &xvar, unsigned int npoints, const double *xvals, const double *yvals, const char *algo) :
        RooAbsReal(name,title),
        xvar_("xvar","Variable", this, xvar), 
        x_(npoints), y_(npoints), type_(algo),
        interp_(0)
{ 
    for (unsigned int i = 0; i < npoints; ++i) {
        x_[i] = xvals[i];
        y_[i] = yvals[i];
    }
}

RooSpline1D::RooSpline1D(const char *name, const char *title, RooAbsReal &xvar, unsigned int npoints, const float *xvals, const float *yvals, const char *algo) :
        RooAbsReal(name,title),
        xvar_("xvar","Variable", this, xvar), 
        x_(npoints), y_(npoints), type_(algo),
        interp_(0)
{ 
    for (unsigned int i = 0; i < npoints; ++i) {
        x_[i] = xvals[i];
        y_[i] = yvals[i];
    }
}


RooSpline1D::~RooSpline1D() 
{
    delete interp_;
}


TObject *RooSpline1D::clone(const char *newname) const 
{
    return new RooSpline1D(newname, this->GetTitle(), const_cast<RooAbsReal &>(xvar_.arg()), x_.size(), &x_[0], &y_[0], type_.c_str());
}

void RooSpline1D::init() const {
    delete interp_;
    if      (type_ == "CSPLINE") interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kCSPLINE);
    else if (type_ == "LINEAR") interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kLINEAR);
    else if (type_ == "POLYNOMIAL") interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kPOLYNOMIAL);
    else if (type_ == "CSPLINE_PERIODIC") interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kCSPLINE_PERIODIC);
    else if (type_ == "AKIMA") interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kAKIMA);
    else if (type_ == "AKIMA_PERIODIC") interp_ = new ROOT::Math::Interpolator(x_, y_, ROOT::Math::Interpolation::kAKIMA_PERIODIC);
    else throw std::invalid_argument("Unknown interpolation type '"+type_+"'");
}

Double_t RooSpline1D::evaluate() const {
    if (interp_ == 0) init();
    return interp_->Eval(xvar_);
}


ClassImp(RooSpline1D)
