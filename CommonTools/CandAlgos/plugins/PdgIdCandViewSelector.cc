/* \class PdgIdCandViewSelector
 * 
 * Candidate Selector based on a pdgId set. The input collection
 * is a View<Candidate>.
 * Saves a collection of references to selected objects
 * Usage:
 * 
 * module leptonRefs = PdgIdCandViewSelector {
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
          reco::CandidateView,
          PdgIdSelector
        > PdgIdCandViewSelector;

DEFINE_FWK_MODULE( PdgIdCandViewSelector );
