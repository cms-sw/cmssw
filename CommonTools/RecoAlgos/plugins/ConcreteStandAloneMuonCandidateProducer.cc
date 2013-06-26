/** \class ConcreteStandAloneMuonCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ConcreteStandAloneMuonCandidateProducer.cc,v 1.2 2013/02/28 00:17:18 wmtan Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/StandAloneMuonTrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::TrackCollection, 
          reco::RecoStandAloneMuonCandidateCollection
        > ConcreteStandAloneMuonCandidateProducer;

DEFINE_FWK_MODULE(ConcreteStandAloneMuonCandidateProducer);
