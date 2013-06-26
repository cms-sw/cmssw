#include "PhysicsTools/PatUtils/interface/SimpleJetTrackAssociator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

void
helper::SimpleJetTrackAssociator::associateTransient(const math::XYZVector &dir, 
                        const reco::TrackCollection &in, reco::TrackRefVector &out) {
        for (size_t i = 0, n = in.size(); i < n; i++) {
                const reco::Track & t = in[i];
                if ((t.numberOfValidHits() < nHits_) || (t.normalizedChi2() > chi2nMax_)) continue;
                if (deltaR2(dir, t) < deltaR2_) {
                        reco::TrackRef tr(&in, i); // note: transient ref
                        out.push_back(tr);
                }
        }
}

void
helper::SimpleJetTrackAssociator::associate(const math::XYZVector &dir, 
                        const edm::View<reco::Track> &in, reco::TrackRefVector &out) {
        for (size_t i = 0, n = in.size(); i < n; i++) {
                const reco::Track & t = in[i];
                if ((t.numberOfValidHits() < nHits_) || (t.normalizedChi2() > chi2nMax_)) continue;
                if (deltaR2(dir, t) < deltaR2_) {
                        reco::TrackRef tr = in.refAt(i).castTo<reco::TrackRef>();
                        out.push_back(tr);
                }
        }
}
