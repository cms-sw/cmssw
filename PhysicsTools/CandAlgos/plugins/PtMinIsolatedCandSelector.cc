/* \class PtMinIsolatedCandSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Usage:
 * 
 * module selectedCands = PtMinIsolatedCandSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 * };
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
          >,
          reco::CandidateCollection
        > PtMinIsolatedCandSelector;

DEFINE_FWK_MODULE( PtMinIsolatedCandSelector );
