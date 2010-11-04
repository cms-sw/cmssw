/* \class CandViewRefSelector
 *
 * PFTau Selector based on a configurable cut.
 * Reads a edm::View<PFTau> as input
 * and saves a vector of references
 * Usage:
 *
 * module selectedCands = PFTauViewRefSelector {
 *   InputTag src = myCollection
 *   string cut = "pt > 15.0"
 * };
 *
 * \author: Luca Lista, INFN
 * \modifications: Evan Friis, UC Davis
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/PFJet.h"

typedef SingleObjectSelector<
          edm::View<reco::Candidate>,
          StringCutObjectSelector<reco::Candidate, true>
       > PFTauViewRefSelector;

DEFINE_FWK_MODULE(PFTauViewRefSelector);

typedef SingleObjectSelector<
          edm::View<reco::Jet>,
          StringCutObjectSelector<reco::Jet, true>
       > JetViewRefSelector;

DEFINE_FWK_MODULE(JetViewRefSelector);
