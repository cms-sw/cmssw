/* \class CandMatchedCloneProducer
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/MatchedCloneProducer.h"

typedef MatchedCloneProducer<reco::CandidateCollection> CandMatchedCloneProducer;

DEFINE_FWK_MODULE( CandMatchedCloneProducer );
