#ifndef RecoJets_FFTJetAlgorithm_JetToPeakDistance_h
#define RecoJets_FFTJetAlgorithm_JetToPeakDistance_h

#include "fftjet/RecombinedJet.hh"

#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"

namespace fftjetcms {
    class JetToPeakDistance
    {
    public:
        explicit JetToPeakDistance(double etaToPhiBandwidthRatio = 1.0);

        double operator()(const fftjet::RecombinedJet<VectorLike>& jet,
                          const fftjet::Peak& peak) const;
    private:
        double etaBw_;
        double phiBw_;
    };
}

#endif // RecoJets_FFTJetAlgorithm_JetToPeakDistance_h
