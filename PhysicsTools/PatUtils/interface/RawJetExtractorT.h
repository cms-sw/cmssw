#ifndef PAT_RAWJETEXTRACTORT_H
#define PAT_RAWJETEXTRACTORT_H

#include "DataFormats/PatCandidates/interface/Jet.h"

namespace pat {

    template <typename T>
    class RawJetExtractorT  {
        public:
        RawJetExtractorT(){}
            reco::Candidate::LorentzVector operator()(const T& jet) const {
                return jet.p4();
            }
    };

    template <>
    class RawJetExtractorT<pat::Jet> {
        public:
            reco::Candidate::LorentzVector operator()(const pat::Jet& jet) const {
                if ( jet.jecSetsAvailable() ) return jet.correctedP4("Uncorrected");
                else return jet.p4();
            }
    };
    
}
#endif
