//=========================================================================
// VBuilders.h
//
// Functors for building various quantities from grid points. They are
// intended for use with "KernelRecombinationAlg" or similar templated
// classes.
//
// I. Volobouev
// March 2010
//=========================================================================

#ifndef RecoJets_FFTJetAlgorithms_VBuilders_h
#define RecoJets_FFTJetAlgorithms_VBuilders_h

#include <cmath>

#include "DataFormats/Math/interface/LorentzVector.h"

namespace fftjetcms {
    struct PtEtaP4Builder
    {
        inline math::XYZTLorentzVector operator()(
            const double pt, const double eta, const double phi) const
            {
                const double px = pt*cos(phi);
                const double py = pt*sin(phi);
                const double pz = pt*sinh(eta);
                const double e = sqrt(px*px + py*py + pz*pz);
                return math::XYZTLorentzVector(px, py, pz, e);
            }
    };

    struct EnergyEtaP4Builder
    {
        inline math::XYZTLorentzVector operator()(
            const double e, const double eta, const double phi) const
            {
                // There is no mass associated with this energy... We will
                // assume that the mass is 0 and proceed as if the energy
                // is the momentum.
                const double pt = e/cosh(eta);
                return math::XYZTLorentzVector(
                    pt*cos(phi), pt*sin(phi), pt*sinh(eta), e);
            }
    };
}

#endif // RecoJets_FFTJetAlgorithms_VBuilders_h
