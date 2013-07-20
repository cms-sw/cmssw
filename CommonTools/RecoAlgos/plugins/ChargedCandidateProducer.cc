/** \class ChargedCandidateProducer
 *
 * Framework module that produces a collection
 * of candidates with a Track Component
 *
 * \author Steven Lowette
 *
 * $Id: ChargedCandidateProducer.cc,v 1.2 2009/11/18 09:12:49 hegner Exp $
 *
 */

#include "CommonTools/RecoAlgos/src/TrackToCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef CandidateProducer<
          edm::View<reco::Track>,
          reco::RecoChargedCandidateCollection,
          AnySelector,
          converter::helper::CandConverter<reco::Track>::type
        > ChargedCandidateProducer;

DEFINE_FWK_MODULE(ChargedCandidateProducer);
