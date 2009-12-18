/** \class ConcreteChargedCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ConcreteChargedCandidateProducer.cc,v 1.1 2009/03/04 13:11:29 llista Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/TrackToRefCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::TrackCollection, 
          reco::RecoChargedRefCandidateCollection
        > ConcreteChargedRefCandidateProducer;

DEFINE_FWK_MODULE(ConcreteChargedRefCandidateProducer);
