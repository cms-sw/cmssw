/* \class PdgIdAndStatusCandDecaySelector
 * 
 * Candidate Selector based on a pdgId set
 * cloning the full decay chain
 * Usage:
 * 
 * module leptons = PdgIdAndStatusCandDecaySelector {
 *   InputTag src = myCollection
 *   vint32 pdgId = { 11, 13 }
 *   vint32 status = { 1 }
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/CandAlgos/interface/CandDecaySelector.h"
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
        > PdgIdAndStatusCandDecaySelector;

DEFINE_FWK_MODULE( PdgIdAndStatusCandDecaySelector );
