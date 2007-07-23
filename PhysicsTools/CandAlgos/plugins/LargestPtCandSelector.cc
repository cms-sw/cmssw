/* \class LargestPtCandSelector
 * 
 * Keep the maxNumber biggest (in respect to Pt) Candidates from myCollection
 * Usage:
 *
 *
 *  module McPartonSele = LargestPtCandSelector {
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
            reco::CandidateCollection,
            GreaterByPt<reco::Candidate>
          >
        > LargestPtCandSelector;

DEFINE_FWK_MODULE( LargestPtCandSelector );
