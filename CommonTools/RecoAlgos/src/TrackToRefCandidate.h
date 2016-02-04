#ifndef RecoAlgos_TrackToRefCandidate_h
#define RecoAlgos_TrackToRefCandidate_h
#include "CommonTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "CommonTools/RecoAlgos/src/CandidateProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace converter {

  struct TrackToRefCandidate : public MassiveCandidateConverter {
    typedef reco::Track value_type;
    typedef reco::TrackCollection Components;
    typedef reco::RecoChargedRefCandidate Candidate;
    TrackToRefCandidate(const edm::ParameterSet & cfg) : 
      MassiveCandidateConverter(cfg) {
    }
    void convert(reco::TrackRef trkRef, reco::RecoChargedRefCandidate & c) const {
      c = reco::RecoChargedRefCandidate( trkRef, sqrt(massSqr_) );
    }  
  };

  namespace helper {
    template<>
    struct CandConverter<reco::Track> { 
      typedef TrackToRefCandidate type;
    };
  }

}

#endif
