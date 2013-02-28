/** \class ConcreteChargedCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ConcreteChargedRefCandidateProducer.cc,v 1.1 2009/11/24 03:47:26 srappocc Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/TrackToRefCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::TrackCollection, 
          reco::RecoChargedRefCandidateCollection
        > ConcreteChargedRefCandidateProducer;

DEFINE_FWK_MODULE(ConcreteChargedRefCandidateProducer);
