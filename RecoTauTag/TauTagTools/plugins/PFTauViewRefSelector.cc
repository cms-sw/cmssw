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
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"

#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PFTauViewRefSelector : public edm::EDFilter {
  public:
    explicit PFTauViewRefSelector(const edm::ParameterSet &pset);
    ~PFTauViewRefSelector() override {}
    bool filter(edm::Event& evt, const edm::EventSetup& es) override;
  private:
    edm::InputTag src_;
    std::string cut_;
    std::unique_ptr<StringCutObjectSelector<reco::PFTau> > outputSelector_;
    bool filter_;
};

PFTauViewRefSelector::PFTauViewRefSelector(const edm::ParameterSet &pset) {
  src_ = pset.getParameter<edm::InputTag>("src");
  std::string cut = pset.getParameter<std::string>("cut");
  filter_ = pset.exists("filter") ? pset.getParameter<bool>("filter") : false;
  outputSelector_ = std::make_unique<StringCutObjectSelector<reco::PFTau>>(cut);
  produces<reco::PFTauRefVector>();
}

bool
PFTauViewRefSelector::filter(edm::Event& evt, const edm::EventSetup& es) {
  auto output = std::make_unique<reco::PFTauRefVector>();
  // Get the input collection to clean
  edm::Handle<reco::CandidateView> input;
  evt.getByLabel(src_, input);
  // Cast the input candidates to Refs to real taus
  reco::PFTauRefVector inputRefs =
      reco::tau::castView<reco::PFTauRefVector>(input);

  for(auto const& tau : inputRefs) {
    if (outputSelector_.get() && (*outputSelector_)(*tau)) {
      output->push_back(tau);
    }
  }
  size_t outputSize = output->size();
  evt.put(std::move(output));
  // Filter if desired and no objects passed our cut
  return !(filter_ && outputSize == 0);
}

DEFINE_FWK_MODULE(PFTauViewRefSelector);

#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/JetReco/interface/PFJet.h"
typedef SingleObjectSelector<
          edm::View<reco::Jet>,
          StringCutObjectSelector<reco::Jet, true>
       > JetViewRefSelector;

DEFINE_FWK_MODULE(JetViewRefSelector);
