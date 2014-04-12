#include <fstream>

#include "RecoJets/FFTJetAlgorithms/interface/EtaDependentPileup.h"

namespace fftjetcms {
    EtaDependentPileup::EtaDependentPileup(
        const fftjet::LinearInterpolator2d& interp,
        const double inputRhoFactor, const double outputRhoFactor)
        : interp_(interp),
          inputRhoFactor_(inputRhoFactor),
          outputRhoFactor_(outputRhoFactor),
          etaMin_(interp.xMin()),
          etaMax_(interp.xMax()),
          rhoMin_(interp.yMin()),
          rhoMax_(interp.yMax()),
          rhoStep_((rhoMax_ - rhoMin_)/interp.ny())
    {
        const double etaStep = (etaMax_ - etaMin_)/interp.nx();
        etaMin_ += etaStep*0.5;
        etaMax_ -= etaStep*0.5;
        rhoMin_ += rhoStep_*0.5;
        rhoMax_ -= rhoStep_*0.5;
        assert(etaMin_ < etaMax_);
        assert(rhoMin_ < rhoMax_);
    }

    double EtaDependentPileup::operator()(
        double eta, double /* phi */,
        const reco::FFTJetPileupSummary& summary) const
    {
        double value = 0.0;
        const double rho = summary.pileupRho()*inputRhoFactor_;
        if (eta < etaMin_)
            eta = etaMin_;
        if (eta > etaMax_)
            eta = etaMax_;
        if (rho >= rhoMin_ && rho <= rhoMax_)
            value = interp_(eta, rho);
        else
        {
            double x0, x1;
            if (rho < rhoMin_)
            {
                x0 = rhoMin_;
                x1 = rhoMin_ + rhoStep_*0.5;
            }
            else
            {
                x0 = rhoMax_;
                x1 = rhoMax_ - rhoStep_*0.5;
            }
            const double z0 = interp_(eta, x0);
            const double z1 = interp_(eta, x1);
            value = z0 + (z1 - z0)*((rho - x0)/(x1 - x0));
        }
        return (value > 0.0 ? value : 0.0)*outputRhoFactor_;
    }
}
