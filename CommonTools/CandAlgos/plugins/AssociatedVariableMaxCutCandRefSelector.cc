/* \class AssociatedVariableMaxCutCandRefSelector
 * 
 * Candidate Selector based on a maximun cut on an
 * associated variable (e.g.: isolation), and saver a 
 * collection of references.
 *
 * Usage:
 * 
 * module selectedCands = AssociatedVariableMaxCutCandRefSelector {
 *   InputTag src = myCollection
 *   double max = 0.2
 * }
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PairSelector.h"
#include "CommonTools/UtilAlgos/interface/RefSelector.h"
#include "CommonTools/UtilAlgos/interface/AnySelector.h"
#include "CommonTools/UtilAlgos/interface/MaxSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef double isolation;

typedef SingleObjectSelector<
          edm::AssociationVector<reco::CandidateRefProd, std::vector<isolation> >,
          PairSelector<
            RefSelector<AnySelector>,
            MaxSelector<isolation>
          >,
          reco::CandidateRefVector
        > AssociatedVariableMaxCutCandRefSelector;

DEFINE_FWK_MODULE( AssociatedVariableMaxCutCandRefSelector );
