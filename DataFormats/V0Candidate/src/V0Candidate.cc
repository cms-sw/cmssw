#include "DataFormats/V0Candidate/interface/V0Candidate.h"

using namespace reco;

V0Candidate::V0Candidate() {
  isGoodV0 = false;
}

V0Candidate::V0Candidate( const VertexCompositeCandidate& theCandIn,
			  const reco::Track& thePosTk,
			  const reco::Track& theNegTk ) {
  theCand = theCandIn;
  posDaughter = thePosTk;
  negDaughter = theNegTk;
  isGoodV0 = true;
}

/*V0Candidate::V0Candidate( const VertexCompositeCandidate& theCandIn,
			  const reco::TrackRef& thePosTkRef,
			  const reco::TrackRef& theNegTkRef ) {
  theCand = theCandIn;
  posDaughter = *thePosTkRef;
  negDaughter = *theNegTkRef;
  isGoodV0 = true;
  }*/
