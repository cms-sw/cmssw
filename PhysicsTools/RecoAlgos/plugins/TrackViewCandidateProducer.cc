/** \class TrackViewCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track compoment from
 * a View<Track>
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: ConcreteChargedCandidateProducer.cc,v 1.1 2007/10/31 15:08:06 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/src/TrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"

typedef CandidateProducer<
          edm::View<reco::Track>,
          reco::RecoChargedCandidateCollection
        > TrackViewCandidateProducer;

DEFINE_FWK_MODULE(TrackViewCandidateProducer);
