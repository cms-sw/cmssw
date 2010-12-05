#include <cmath>
#include <cassert>

#include "RecoJets/FFTJetAlgorithms/interface/JetConvergenceDistance.h"

namespace fftjetcms {
    JetConvergenceDistance::JetConvergenceDistance(
        const double etaToPhiBandwidthRatio,
        const double relativePtBandwidth)
        : etaBw_(sqrt(etaToPhiBandwidthRatio)),
          phiBw_(1.0/etaBw_),
          ptBw_(relativePtBandwidth)
    {
        assert(etaToPhiBandwidthRatio > 0.0);
        assert(relativePtBandwidth > 0.0);
    }

    double JetConvergenceDistance::operator()(
        const fftjet::RecombinedJet<VectorLike>& j1,
        const fftjet::RecombinedJet<VectorLike>& j2) const
    {
        const double deta = (j1.vec().Eta() - j2.vec().Eta())/etaBw_;
        double dphi = j1.vec().Phi() - j2.vec().Phi();
        if (dphi > M_PI)
            dphi -= (2.0*M_PI);
        else if (dphi < -M_PI)
            dphi += (2.0*M_PI);
        dphi /= phiBw_;
        const double mag1 = j1.magnitude();
        const double mag2 = j2.magnitude();
        double dmag = 0.0;
        if (mag1 > 0.0 || mag2 > 0.0)
            dmag = 2.0*(mag1 - mag2)/(mag1 + mag2)/ptBw_;
        return sqrt(deta*deta + dphi*dphi + dmag*dmag);
    }
}
