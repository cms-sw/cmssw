/* \class AssociatedVariableMaxCutCandViewSelector
 * 
 * Candidate Selector based on a maximun cut on an
 * associated variable (e.g.: isolation), and saver a 
 * collection of references.
 *
 * Usage:
 * 
 * module selectedCands = AssociatedVariableMaxCutCandViewSelector {
 *   InputTag src = myCollection
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
#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/MaxSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
typedef double isolation;

typedef SingleObjectSelector<
          reco::CandViewDoubleAssociations,
          PairSelector<
            RefSelector<AnySelector>,
            MaxSelector<isolation>
          >
        > AssociatedVariableMaxCutCandViewSelector;

DEFINE_FWK_MODULE( AssociatedVariableMaxCutCandViewSelector );
