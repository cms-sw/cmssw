#ifndef RecoAlgos_TrackToCandidate_h
#define RecoAlgos_TrackToCandidate_h
#include "PhysicsTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "PhysicsTools/RecoAlgos/src/CandidateProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"

namespace converter {

  struct TrackToCandidate : public MassiveCandidateConverter {
    typedef reco::TrackCollection Components;
    typedef reco::RecoChargedCandidate Candidate;
    TrackToCandidate( const edm::ParameterSet & cfg ) : 
      MassiveCandidateConverter( cfg ) {
    }
    void convert( size_t idx, const edm::Handle<reco::TrackCollection> & , 
		  reco::RecoChargedCandidate & ) const;
  };

  namespace helper {
    template<>
    struct CandConverter<reco::TrackCollection> { 
      typedef TrackToCandidate type;
    };
  }

}

#endif
