/* \class RefCandViewRefSelector
 * 
 * RefCandidate Selector based on a configurable cut.
 * Reads a edm::View<RefCandidate> as input
 * and saves a vector of references
 * Usage:
 * 
 * module selectedCands = RefCandViewRefSelector {
 *   InputTag src = myCollection
 *   string cut = "pt > 15.0 & track().isValid()"
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

typedef SingleObjectSelector<
          edm::View<reco::RecoCandidate>,
          StringCutObjectSelector<reco::RecoCandidate>
       > RecoCandViewRefSelector;

DEFINE_FWK_MODULE(RecoCandViewRefSelector);
