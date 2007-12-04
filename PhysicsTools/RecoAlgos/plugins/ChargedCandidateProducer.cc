/** \class ChargedCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ChargedCandidateProducer.cc,v 1.1 2007/02/01 13:11:18 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/src/TrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          reco::TrackCollection, 
          reco::CandidateCollection
        > ChargedCandidateProducer;

DEFINE_FWK_MODULE( ChargedCandidateProducer );
