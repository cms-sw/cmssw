/* \class PtMinCandSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Usage:
 * 
 * module selectedCands = PtMinCandSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

typedef ObjectSelector<
          SingleElementCollectionSelector<
            reco::CandidateCollection,
            PtMinSelector<reco::Candidate>
          >
        > PtMinCandSelector;

DEFINE_FWK_MODULE( PtMinCandSelector );
