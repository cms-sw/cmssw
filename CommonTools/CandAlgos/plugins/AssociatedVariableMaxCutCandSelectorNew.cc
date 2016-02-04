/* \class AssociatedVariableMaxCutCandSelectorNew
 * 
 * Candidate Selector based on a maximun cut on an
 * associated variable (e.g.: isolation), and saver a 
 * collection of references.
 *
 * Usage:
 * 
 * module selectedCands = AssociatedVariableMaxCutCandSelectorNew {
 *   InputTag src = myCollection
 *   InputTag var = myVariable
 *   double max = 0.2
 * }
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/AssociatedVariableCollectionSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/RefSelector.h"
#include "CommonTools/UtilAlgos/interface/AnySelector.h"
#include "CommonTools/UtilAlgos/interface/MaxSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"


typedef ObjectSelector<
          AssociatedVariableCollectionSelector<
            reco::CandidateView, edm::ValueMap<float>,
            AndSelector<
              AnySelector,
              MaxSelector<float>
            >
          >
        > AssociatedVariableMaxCutCandSelectorNew;

DEFINE_FWK_MODULE( AssociatedVariableMaxCutCandSelectorNew );
