//=========================================================================
// AbsPileupCalculator.h
//
// Interface for calculators of the pile-up density
//
// I. Volobouev
// June 2011
//=========================================================================

#ifndef RecoJets_FFTJetAlgorithms_AbsPileupCalculator_h
#define RecoJets_FFTJetAlgorithms_AbsPileupCalculator_h

#include "DataFormats/JetReco/interface/FFTJetPileupSummary.h"

namespace fftjetcms {
    struct AbsPileupCalculator
    {
        virtual ~AbsPileupCalculator() {}

        virtual double operator()(
            double eta, double phi,
            const reco::FFTJetPileupSummary& summary) const = 0;

        virtual bool isPhiDependent() const = 0;
    };
}

#endif // RecoJets_FFTJetAlgorithms_AbsPileupCalculator_h
