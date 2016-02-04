/* \class PdgIdAndStatusCandViewSelector
 * 
 * Candidate Selector based on a pdgId set and status.
 * The input collection is a View<Candidate>.
 * Saves a collection of references to the selected objects.
 *
 * Usage:
 * 
 * module leptonRefs = PdgIdAndStatusCandViewSelector {
 *   InputTag src = myCollection
 *   vint32 pdgId = { 11, 13 }
 *   vint32 status =  { 1 }
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "CommonTools/UtilAlgos/interface/PdgIdSelector.h"
#include "CommonTools/UtilAlgos/interface/StatusSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          reco::CandidateView,
          AndSelector<
            PdgIdSelector,
            StatusSelector
          >
        > PdgIdAndStatusCandViewSelector;

DEFINE_FWK_MODULE( PdgIdAndStatusCandViewSelector );
