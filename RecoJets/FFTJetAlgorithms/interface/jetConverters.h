#ifndef RecoJets_FFTJetAlgorithms_jetConverters_h
#define RecoJets_FFTJetAlgorithms_jetConverters_h

#include "fftjet/RecombinedJet.hh"

#include "DataFormats/JetReco/interface/FFTJet.h"
#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"

namespace fftjetcms {
    // The function below makes a storable FFTJet
    template<class Real>
    reco::FFTJet<Real> jetToStorable(
        const fftjet::RecombinedJet<VectorLike>& jet);

    // The function below restores the original RecombinedJet object
    template<class Real>
    fftjet::RecombinedJet<VectorLike> jetFromStorable(
        const reco::FFTJet<Real>& jet);
}

////////////////////////////////////////////////////////////////////////
//
//  Implementation follows
//
////////////////////////////////////////////////////////////////////////

namespace fftjetcms {
    template<class Real>
    reco::FFTJet<Real> jetToStorable(
        const fftjet::RecombinedJet<VectorLike>& jet)
    {
        typedef reco::PattRecoPeak<Real> StoredPeak;

        double hessian[3] = {0., 0., 0.};
        const fftjet::Peak& peak(jet.precluster());
        peak.hessian(hessian);

        return reco::FFTJet<Real>(StoredPeak(peak.eta(),
                                             peak.phi(),
                                             peak.magnitude(),
                                             hessian,
                                             peak.driftSpeed(),
                                             peak.magSpeed(),
                                             peak.lifetime(),
                                             peak.scale(),
                                             peak.nearestNeighborDistance(),
                                             peak.clusterRadius(),
                                             peak.clusterSeparation()),
                                  jet.vec(), jet.ncells(), jet.etSum(),
                                  jet.centroidEta(), jet.centroidPhi(),
                                  jet.etaWidth(), jet.phiWidth(),
                                  jet.etaPhiCorr(), jet.fuzziness(),
                                  jet.convergenceDistance(),
                                  jet.recoScale(), jet.recoScaleRatio(),
                                  jet.membershipFactor(),
                                  jet.code(), jet.status());
    }


    template<class Real>
    fftjet::RecombinedJet<VectorLike> jetFromStorable(
        const reco::FFTJet<Real>& jet)
    {
        typedef reco::PattRecoPeak<Real> StoredPeak;
        typedef fftjet::RecombinedJet<VectorLike> RecoFFTJet;

        double hessian[3] = {0., 0., 0.};
        const StoredPeak& p(jet.f_precluster());
        p.hessian(hessian);

        return RecoFFTJet(fftjet::Peak(p.eta(), p.phi(), p.magnitude(),
                                       hessian, p.driftSpeed(),
                                       p.magSpeed(), p.lifetime(),
                                       p.scale(), p.nearestNeighborDistance(),
                                       jet.f_membershipFactor(),
                                       jet.f_recoScale(),
                                       jet.f_recoScaleRatio(),
                                       p.clusterRadius(),
                                       p.clusterSeparation(), jet.f_code(),
                                       jet.f_status()),
                          jet.f_vec(), jet.f_ncells(), jet.f_etSum(),
                          jet.f_centroidEta(), jet.f_centroidPhi(),
                          jet.f_etaWidth(), jet.f_phiWidth(),
                          jet.f_etaPhiCorr(), jet.f_fuzziness(),
                          0.0, 0.0, jet.f_convergenceDistance());
    }
}

#endif // RecoJets_FFTJetAlgorithms_jetConverters_h
