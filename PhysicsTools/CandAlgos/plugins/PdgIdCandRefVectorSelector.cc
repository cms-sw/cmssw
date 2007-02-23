/* \class PdgIdCandRefVectorSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Usage:
 * 
 * module selectedCands = PdgIdCandSelector {
 *   InputTag src = myCollection
 *   vint32 pdgId = { 15.0
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectRefVectorSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PdgIdSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

typedef ObjectRefVectorSelector <
          SingleElementCollectionSelector<
            reco::CandidateCollection,
            PdgIdSelector<reco::Candidate>,
            reco::CandidateRefVector
          >
        > PdgIdCandRefVectorSelector;

DEFINE_FWK_MODULE( PdgIdCandRefVectorSelector );
