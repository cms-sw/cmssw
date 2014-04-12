/* \class PdgIdCandSelector
 * 
 * Candidate Selector based on a pdgId set
 * Usage:
 * 
 * module leptons = PdgIdCandSelector {
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

typedef SingleObjectSelector<
          reco::CandidateCollection,
          PdgIdSelector
        > PdgIdCandSelector;

DEFINE_FWK_MODULE( PdgIdCandSelector );
