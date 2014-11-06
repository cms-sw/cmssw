#include "PhysicsTools/Heppy/interface/CMGMuonCleanerBySegmentsAlgo.h"

namespace heppy {

CMGMuonCleanerBySegmentsAlgo::~CMGMuonCleanerBySegmentsAlgo() {
}

std::vector<bool> 
CMGMuonCleanerBySegmentsAlgo::clean(const std::vector<pat::Muon> &src) const {
    unsigned int nsrc = src.size();
    std::vector<bool> good(nsrc, true);
    for (unsigned int i = 0; i < nsrc; ++i) {
        const pat::Muon &mu1    = src[i];
        if (!preselection_(mu1)) good[i] = false; 
        if (!good[i]) continue;
        int  nSegments1 = mu1.numberOfMatches(reco::Muon::SegmentArbitration);
        for (unsigned int j = i+1; j < nsrc; ++j) {
            const pat::Muon &mu2    = src[j];
            if (isSameMuon(mu1,mu2)) continue;
            if (!good[j] || !preselection_(mu2)) continue;
            int nSegments2 = mu2.numberOfMatches(reco::Muon::SegmentArbitration);
            if (nSegments2 == 0 || nSegments1 == 0) continue;
            double sf = muon::sharedSegments(mu1,mu2)/std::min<double>(nSegments1,nSegments2);
            if (sf > sharedFraction_) {
                if (isBetterMuon(mu1,mu1.isPFMuon(),mu2,mu2.isPFMuon())) {
                    good[j] = false;
                } else {
                    good[i] = false;
                }
            }
        }
    }

    for (unsigned int i = 0; i < nsrc; ++i) {
        const pat::Muon &mu1    = src[i];
        if (passthrough_(mu1)) good[i] = true;
    }

    return good;
}

bool
CMGMuonCleanerBySegmentsAlgo::isSameMuon(const pat::Muon &mu1, const pat::Muon &mu2) const {
    return (& mu1 == & mu2)  ||
           (mu1.originalObjectRef() == mu2.originalObjectRef()) ||
           (mu1.reco::Muon::innerTrack().isNonnull() ?
                   mu1.reco::Muon::innerTrack() == mu2.reco::Muon::innerTrack() :
                   mu1.reco::Muon::outerTrack() == mu2.reco::Muon::outerTrack());
}

bool
CMGMuonCleanerBySegmentsAlgo::isBetterMuon(const pat::Muon &mu1, bool mu1PF, const pat::Muon &mu2, bool mu2PF) const {
    if (mu2.track().isNull()) return true;
    if (mu1.track().isNull()) return false;
    if (mu1PF != mu2PF) return mu1PF;
    if (mu1.isGlobalMuon() != mu2.isGlobalMuon()) return mu1.isGlobalMuon();
    if (mu1.charge() == mu2.charge() && deltaR2(mu1,mu2) < 0.0009) {
        return mu1.track()->ptError()/mu1.track()->pt() < mu2.track()->ptError()/mu2.track()->pt();
    } else {
        int nm1 = mu1.numberOfMatches(reco::Muon::SegmentArbitration);
        int nm2 = mu2.numberOfMatches(reco::Muon::SegmentArbitration);
        return (nm1 != nm2 ? nm1 > nm2 : mu1.pt() > mu2.pt());
    }
}
}
