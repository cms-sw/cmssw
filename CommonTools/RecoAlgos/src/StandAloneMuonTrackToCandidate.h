#ifndef RecoAlgos_StandAloneMuonTrackToCandidate_h
#define RecoAlgos_StandAloneMuonTrackToCandidate_h
#include "CommonTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "CommonTools/RecoAlgos/src/CandidateProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoStandAloneMuonCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoStandAloneMuonCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace converter {

  struct StandAloneMuonTrackToCandidate : public MassiveCandidateConverter {
    typedef reco::Track value_type;
    typedef reco::TrackCollection Components;
    typedef reco::RecoStandAloneMuonCandidate Candidate;
    StandAloneMuonTrackToCandidate(const edm::ParameterSet & cfg) : 
      MassiveCandidateConverter(cfg) {
    }
    void convert(reco::TrackRef trkRef, reco::RecoStandAloneMuonCandidate & c) const {
      const reco::Track & trk = * trkRef;
      c.setCharge(trk.charge());
      c.setVertex(trk.vertex());
      reco::Track::Vector p = trk.momentum();
      double t = sqrt(massSqr_ + p.mag2());
      c.setP4(reco::Candidate::LorentzVector(p.x(), p.y(), p.z(), t));
      c.setTrack(trkRef);
      c.setPdgId(particle_.pdgId());
    }  
  };

  namespace helper {
    template<>
    struct CandConverter<reco::Track> { 
      typedef StandAloneMuonTrackToCandidate type;
    };
  }

}

#endif
