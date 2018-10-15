/* \class PFTauViewRefMerger
 *
 * Produces a RefVector of PFTaus from a different views of PFTau.
 *
 * Note that the collections must all come from the same original collection!
 *
 * Author: Evan K. Friis
 *
 */

#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

class PFTauViewRefMerger : public edm::EDProducer {
  public:
    explicit PFTauViewRefMerger(const edm::ParameterSet& cfg) :
        src_(cfg.getParameter<std::vector<edm::InputTag> >("src")) {
          produces<reco::PFTauRefVector>();
        }
  private:
    void produce(edm::Event & evt, const edm::EventSetup &) override {
      auto out = std::make_unique<reco::PFTauRefVector>();
      for(auto const& inputSrc : src_) {
        edm::Handle<reco::CandidateView> src;
        evt.getByLabel(inputSrc, src);
        reco::PFTauRefVector inputRefs =
            reco::tau::castView<reco::PFTauRefVector>(src);
        // Merge all the collections
        for(auto const& tau : inputRefs) {
          out->push_back(tau);
        }
      }
      evt.put(std::move(out));
    }
    std::vector<edm::InputTag> src_;
};

DEFINE_FWK_MODULE(PFTauViewRefMerger);
