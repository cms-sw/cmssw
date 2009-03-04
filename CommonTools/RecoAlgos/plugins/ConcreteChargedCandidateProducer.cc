/** \class ConcreteChargedCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: ConcreteChargedCandidateProducer.cc,v 1.2 2007/12/03 15:38:49 llista Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/TrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::TrackCollection, 
          reco::RecoChargedCandidateCollection
        > ConcreteChargedCandidateProducer;

DEFINE_FWK_MODULE(ConcreteChargedCandidateProducer);
