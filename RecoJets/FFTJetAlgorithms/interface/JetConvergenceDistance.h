#ifndef RecoJets_FFTJetAlgorithm_JetConvergenceDistance_h
#define RecoJets_FFTJetAlgorithm_JetConvergenceDistance_h

#include "fftjet/SimpleFunctors.hh"
#include "fftjet/RecombinedJet.hh"

#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"

namespace fftjetcms {
    class JetConvergenceDistance : public fftjet::Functor2 <
        double,
        fftjet::RecombinedJet<VectorLike>,
        fftjet::RecombinedJet<VectorLike> >
    {
    public:
        JetConvergenceDistance(double etaToPhiBandwidthRatio,
                               double relativePtBandwidth);

        double operator()(const fftjet::RecombinedJet<VectorLike>& jet1,
                          const fftjet::RecombinedJet<VectorLike>& jet2) const override;
    private:
        JetConvergenceDistance() = delete;

        double etaBw_;
        double phiBw_;
        double ptBw_;
    };
}

#endif // RecoJets_FFTJetAlgorithm_JetConvergenceDistance_h
