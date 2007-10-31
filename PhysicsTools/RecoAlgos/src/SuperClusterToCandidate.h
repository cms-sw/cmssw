#ifndef RecoAlgos_SuperClusterToCandidate_h
#define RecoAlgos_SuperClusterToCandidate_h
#include "PhysicsTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "PhysicsTools/RecoAlgos/src/CandidateProducer.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"

namespace converter {
  struct SuperClusterToCandidate : public MassiveCandidateConverter {
    typedef reco::SuperClusterCollection Components;
    typedef reco::RecoEcalCandidate Candidate;
    SuperClusterToCandidate( const edm::ParameterSet & cfg ) : 
      MassiveCandidateConverter( cfg ) {
    }
    void convert( size_t idx, const edm::Handle<reco::SuperClusterCollection> & , 
		  reco::RecoEcalCandidate & ) const;
  };

  namespace helper {
    template<>
    struct CandConverter<reco::SuperClusterCollection> { 
      typedef SuperClusterToCandidate type;
    };
  }
}

#endif
