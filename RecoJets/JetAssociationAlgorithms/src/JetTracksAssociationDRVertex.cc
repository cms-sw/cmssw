// Associate jets with tracks by simple "dR" criteria
// Fedor Ratnikov (UMd), Aug. 28, 2007

#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"

JetTracksAssociationDRVertex::JetTracksAssociationDRVertex(double fDr) : mDeltaR2Threshold(fDr * fDr) {}

void JetTracksAssociationDRVertex::produce(reco::JetTracksAssociation::Container* fAssociation,
                                           const std::vector<edm::RefToBase<reco::Jet> >& fJets,
                                           const std::vector<reco::TrackRef>& fTracks) const {
  // cache tracks kinematics
  std::vector<math::RhoEtaPhiVector> trackP3s;
  trackP3s.reserve(fTracks.size());
  for (const auto& fTrack : fTracks) {
    const reco::Track* track = &*fTrack;
    trackP3s.push_back(math::RhoEtaPhiVector(track->p(), track->eta(), track->phi()));
  }
  //loop on jets and associate
  for (const auto& fJet : fJets) {
    reco::TrackRefVector assoTracks;
    const reco::Jet* jet = &*fJet;
    double jetEta = jet->eta();
    double jetPhi = jet->phi();
    for (unsigned t = 0; t < fTracks.size(); ++t) {
      double dR2 = deltaR2(jetEta, jetPhi, trackP3s[t].eta(), trackP3s[t].phi());
      if (dR2 < mDeltaR2Threshold)
        assoTracks.push_back(fTracks[t]);
    }
    reco::JetTracksAssociation::setValue(fAssociation, fJet, assoTracks);
  }
}
