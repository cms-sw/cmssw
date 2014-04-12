/* \class PdgIdCandRefSelector
 * 
 * Candidate Selector based on a pdgId set.
 * Saves a collection of references to selected objects
 * Usage:
 * 
 * module leptonRefs = PdgIdCandRefSelector {
 *   InputTag src = myCollection
 *   vint32 pdgId = { 11, 13 }
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PdgIdSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector <
          reco::CandidateCollection,
          PdgIdSelector,
          reco::CandidateRefVector
        > PdgIdCandRefSelector;

DEFINE_FWK_MODULE( PdgIdCandRefSelector );
