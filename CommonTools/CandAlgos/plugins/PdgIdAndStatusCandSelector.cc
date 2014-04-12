/* \class PdgIdAndStatusCandSelector
 * 
 * Candidate Selector based on a pdgId set
 * and status.
 * Usage:
 * 
 * module leptons = PdgIdAndStatusCandSelector {
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
          reco::CandidateCollection,
          AndSelector<
            PdgIdSelector,
            StatusSelector
          >
        > PdgIdAndStatusCandSelector;

DEFINE_FWK_MODULE( PdgIdAndStatusCandSelector );
