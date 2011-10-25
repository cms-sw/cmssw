#include <cmath>
#include <cassert>

#include "RecoJets/FFTJetAlgorithms/interface/JetToPeakDistance.h"

namespace fftjetcms {
    JetToPeakDistance::JetToPeakDistance(const double etaToPhiBandwidthRatio)
        : etaBw_(sqrt(etaToPhiBandwidthRatio)),
          phiBw_(1.0/etaBw_)
    {
        assert(etaToPhiBandwidthRatio > 0.0);
    }

    double JetToPeakDistance::operator()(
        const fftjet::RecombinedJet<VectorLike>& j1,
        const fftjet::Peak& peak) const
    {
        if (peak.membershipFactor() <= 0.0)
            // This peak essentially does not exist...
            return 2.0e300;

        const double deta = (j1.vec().Eta() - peak.eta())/etaBw_;
        double dphi = j1.vec().Phi() - peak.phi();
        if (dphi > M_PI)
            dphi -= (2.0*M_PI);
        else if (dphi < -M_PI)
            dphi += (2.0*M_PI);
        dphi /= phiBw_;
        return sqrt(deta*deta + dphi*dphi);
    }
}
