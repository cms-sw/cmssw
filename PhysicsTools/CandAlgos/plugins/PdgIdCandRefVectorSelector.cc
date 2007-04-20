/* \class PdgIdCandRefVectorSelector
 * 
 * Candidate Selector based on a pdgId set.
 * Saves a collection of references to selected objects
 * Usage:
 * 
 * module leptonRefs = PdgIdCandRefVectorSelector {
 *   InputTag src = myCollection
 *   vint32 pdgId = { 11, 13 }
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PdgIdSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector <
            reco::CandidateCollection,
            PdgIdSelector<reco::Candidate>,
            edm::RefVector<reco::CandidateCollection>
        > PdgIdCandRefVectorSelector;

DEFINE_FWK_MODULE( PdgIdCandRefVectorSelector );
