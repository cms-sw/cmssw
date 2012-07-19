// Common tool for pileup subtraction. Can be used
// to subtract pileup as a 4-vector or just to scale
// the jet 4-vector so that its Pt becomes less by
// the amount of pileup Pt.
//
// If the amount of pileup is larger than the amount
// of Pt available in the jet, all components of the
// jet 4-vector will be set to 0.
//
// I. Volobouev
// July 2012

#ifndef RecoJets_FFTJetAlgorithms_adjustForPileup_h
#define RecoJets_FFTJetAlgorithms_adjustForPileup_h

#include "DataFormats/Math/interface/LorentzVector.h"

namespace fftjetcms {
    math::XYZTLorentzVector adjustForPileup(
        const math::XYZTLorentzVector& jet,
        const math::XYZTLorentzVector& pileup,
        bool subtractPileupAs4Vec);
}

#endif // RecoJets_FFTJetAlgorithms_adjustForPileup_h
