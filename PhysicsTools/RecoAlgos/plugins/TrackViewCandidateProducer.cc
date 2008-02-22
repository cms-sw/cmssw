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
 * $Id: TrackViewCandidateProducer.cc,v 1.1 2007/12/03 15:38:49 llista Exp $
 *
 */

#include "PhysicsTools/RecoAlgos/src/TrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"

typedef CandidateProducer<
          edm::View<reco::Track>,
          reco::RecoChargedCandidateCollection,
          StringCutObjectSelector<reco::Track>
        > TrackViewCandidateProducer;

DEFINE_FWK_MODULE(TrackViewCandidateProducer);
