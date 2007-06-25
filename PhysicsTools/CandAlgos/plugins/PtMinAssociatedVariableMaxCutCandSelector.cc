/* \class PtMinAssociatedVariableMaxCutCandSelector
 * 
 * Candidate Selector based on a minimum pt cut
 * plus a cut on an associated variable (e.g.: isolation)
 *
 * Usage:
 * 
 * module selectedCands = PtMinAssociatedVariableMaxCutCandSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 *   double max = 0.2
 * }
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/RefSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MaxSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef double isolation;

typedef SingleObjectSelector<
          edm::AssociationVector<reco::CandidateRefProd, std::vector<isolation> >,
          PairSelector<
            RefSelector<PtMinSelector<reco::Candidate> >,
            MaxSelector<isolation>
          >
        > PtMinAssociatedVariableMaxCutCandSelector;

DEFINE_FWK_MODULE( PtMinAssociatedVariableMaxCutCandSelector );
