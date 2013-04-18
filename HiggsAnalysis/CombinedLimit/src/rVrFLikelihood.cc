#include "../interface/rVrFLikelihood.h"
#include <cstdio>

rVrFLikelihood::rVrFLikelihood(const char *name, const char *title) :
        RooAbsReal(name,title)
{ 
}

void
rVrFLikelihood::addChannel(const TH2* chi2, RooAbsReal &muV, RooAbsReal &muF) 
{
    channels_.push_back(Channel(this,chi2,muV,muF));
}

rVrFLikelihood::~rVrFLikelihood() 
{
}

TObject 
*rVrFLikelihood::clone(const char *newname) const 
{
    // never understood if RooFit actually cares of const-correctness or not.
    rVrFLikelihood* ret = new rVrFLikelihood(newname, this->GetTitle());
    for (std::vector<Channel>::const_iterator it = channels_.begin(), ed = channels_.end(); it != ed; ++it) {
        ret->addChannel(it->chi2, const_cast<RooAbsReal &>(it->muV.arg()), const_cast<RooAbsReal &>(it->muF.arg()));
    }
    return ret;
}

Double_t rVrFLikelihood::evaluate() const {
    double ret = 0;
    for (std::vector<Channel>::const_iterator it = channels_.begin(), ed = channels_.end(); it != ed; ++it) {
        double x = it->muF;
        double y = it->muV;
        if (!(x > it->chi2->GetXaxis()->GetXmin())) return 9999;
        if (!(x < it->chi2->GetXaxis()->GetXmax())) return 9999;
        if (!(y > it->chi2->GetYaxis()->GetXmin())) return 9999;
        if (!(y < it->chi2->GetYaxis()->GetXmax())) return 9999;
        /*printf("looked up %s at (%g,%g), x [%g,%g], y [%g,%g]\n",
                    it->chi2->GetName(), x, y,
                    it->chi2->GetXaxis()->GetXmin(), it->chi2->GetXaxis()->GetXmax(),
                    it->chi2->GetYaxis()->GetXmin(), it->chi2->GetYaxis()->GetXmax());*/
        ret += (const_cast<TH2 *>(it->chi2))->Interpolate(x,y); // WTF Interpolate is not const??
    }
    return 0.5*ret; // NLL
}

ClassImp(rVrFLikelihood)
