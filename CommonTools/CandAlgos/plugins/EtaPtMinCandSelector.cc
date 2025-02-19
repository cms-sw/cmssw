/* \class EtaPtMinCandSelector
 * 
 * Candidate Selector based on a minimum pt cut and an eta range.
 * Usage:
 * 
 * module selectedCands = EtaPtMinCandSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 *   double etaMin = -2
 *   double etaMax = 2
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/EtaRangeSelector.h"
#include "CommonTools/UtilAlgos/interface/AndSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          reco::CandidateCollection,
          AndSelector<
            PtMinSelector,
            EtaRangeSelector
          >
        > EtaPtMinCandSelector;

DEFINE_FWK_MODULE( EtaPtMinCandSelector );
