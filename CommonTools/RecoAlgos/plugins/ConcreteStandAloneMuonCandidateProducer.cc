/** \class ConcreteStandAloneMuonCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ConcreteStandAloneMuonCandidateProducer.cc,v 1.1 2007/12/14 12:49:28 llista Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/StandAloneMuonTrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::TrackCollection, 
          reco::RecoStandAloneMuonCandidateCollection
        > ConcreteStandAloneMuonCandidateProducer;

DEFINE_FWK_MODULE(ConcreteStandAloneMuonCandidateProducer);
