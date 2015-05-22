#include "NeutrinoFitters.h"
#include <cmath>
#include <cstdio>

using cmg::BaseNeutrinoFitter;

BaseNeutrinoFitter::BaseNeutrinoFitter() :
    minimizer_(ROOT::Minuit2::kMigrad),
    functor_(this, &BaseNeutrinoFitter::eval, 4)
{
    minimizer_.SetMaxFunctionCalls(1000000);
    minimizer_.SetMaxIterations(100000);
    minimizer_.SetTolerance(0.001);
    minimizer_.SetFunction(functor_);
}

void BaseNeutrinoFitter::initMET(double met, double metphi, double htJet25)
{
    metx_ = met*std::cos(metphi);
    mety_ = met*std::sin(metphi);
    metsig2_ = 350 + 0.42*met + 0.0004*met*met;
}

double BaseNeutrinoFitter::metNll() const
{
    return (std::pow(metx_ - nu_.Px() - nubar_.Px(), 2) + std::pow(mety_ - nu_.Py() - nubar_.Py(), 2))/metsig2_;     
}

void BaseNeutrinoFitter::setWlnu(TLorentzVector &w, TLorentzVector &nu, const TLorentzVector &l, double costheta, double phi)
{
    // mw^2 = 2 pnu plep (1 - cos(theta))
    double pnu = (0.5*80.4*80.4) / ( l.P() * std::max(0.001, 1.0 - costheta) );
    double sintheta = std::sqrt(1.0 - costheta*costheta);
    // first define the nu in the frame where the lepton has momentum aligned along the Z axis
    nu.SetXYZT(pnu*sintheta*std::cos(phi), pnu*sintheta*std::sin(phi), pnu*costheta, pnu);
    //TLorentzVector lz(0,0,l.P(),l.E());
    //printf("%.3f \n", std::sqrt(2*lz.P()*pnu*(1-costheta))); // --> 80.4
    //printf("%.3f \n", (lz+nu).M()); // --> 80.4
    nu.RotateY(l.Theta());
    //lz.RotateY(l.Theta());
    //printf("%.3f (lz theta %+.2f, l theta %+.2f)\n", (l+nu).M(), lz.Theta(), l.Theta()); // --> 80.4 and the two thetas are the same
    nu.RotateZ(l.Phi());
    w = nu + l;
}

double BaseNeutrinoFitter::nll(double costheta, double phi, double costhetabar, double phibar) 
{
    setWlnu(wp_, nu_,    lp_, costheta,    phi);
    setWlnu(wm_, nubar_, lm_, costhetabar, phibar);
    return metNll();
}

void BaseNeutrinoFitter::initLep(const TLorentzVector &lp, const TLorentzVector &lm) 
{ 
    lp_ = lp; lm_ = lm; 
}

void BaseNeutrinoFitter::fit() 
{
    // initialize variables
    minimizer_.SetLimitedVariable(0, "costheta",    0., 0.01,  -1.,     1.);
    minimizer_.SetLimitedVariable(1, "phi",         0., 0.01,  -3.1416, 3.1416);
    minimizer_.SetLimitedVariable(2, "costhetabar", 0., 0.01,  -1.,     1.);
    minimizer_.SetLimitedVariable(3, "phibar",      0., 0.01,  -3.1416, 3.1416);

    // minimize nll
    minimizer_.Minimize();

    // re-evaluate nll to make sure all 4-vectors are set
    const double *xs = minimizer_.X(); 
    nll(xs[0],xs[1],xs[2],xs[3]);
    //printf("costheta = %+.3f  phi = %+.3f   costhetabar = %+.3f  phi = %+.3f\n", xs[0],xs[1],xs[2],xs[3]);
}

using cmg::TwoMTopNeutrinoFitter;

double TwoMTopNeutrinoFitter::nll(double costheta, double phi, double costhetabar, double phibar) 
{
    //printf("costheta = %+.3f  phi = %+.3f   costhetabar = %+.3f  phi = %+.3f\n", costheta,phi, costhetabar,phibar);
    double nll0 = BaseNeutrinoFitter::nll(costheta,phi, costhetabar,phibar);
    //printf("nll0 = %+8.4f, mW+ = %8.2f mW- = %8.2f\n", nll0, wp_.M(), wm_.M());
    t_    = b_    + wp_;    
    tbar_ = bbar_ + wm_;    
    return nll0 + topMassNll(t_.M())  + topMassNll(tbar_.M());
}

double TwoMTopNeutrinoFitter::topMassNll(double mass) const 
{
    mass -= mTopOffset_;
    //if (mass < 115) mass = 115;
    //if (mass > 205) mass = 205;
    double norm[3]   = { 240.,    85.,   7.  };
    double mu[3]     = { 173.3,  170., 146.  };
    double sigma[3]  = {   6.1,   12.,  11.3 }; 
    double pdf = norm[0]*std::exp(-0.5*std::pow((mass-mu[0])/sigma[0], 2)) + 
                 norm[1]*std::exp(-0.5*std::pow((mass-mu[1])/sigma[1], 2)) + 
                 norm[2]*std::exp(-0.5*std::pow((mass-mu[2])/sigma[2], 2));
    //printf("mtop_pdf(%6.2f) = %9g  (NLL = %+8.4f)\n", mass, pdf, -std::log(0.001*pdf));
    return -std::log(0.001*pdf);
}
