/* \class LargestPtCandViewSelector
 * 
 * Keep a fixed number of largest pt candidates.
 * The input collection is read as View<Candidate>.
 * Saves a collection of references to selected objects.
 * Usage:
 *
 *
 *  module McPartonRefs = LargestPtCandViewSelector {
 *      InputTag src     = myCollection
 *      uint32 maxNumber = 3		
 * } 
 *
 * \author: Loic Quertenmont, UCL
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef ObjectSelector<
          SortCollectionSelector<
            reco::CandidateView,
            GreaterByPt<reco::Candidate>
          >
        > LargestPtCandViewSelector;

DEFINE_FWK_MODULE( LargestPtCandViewSelector );
