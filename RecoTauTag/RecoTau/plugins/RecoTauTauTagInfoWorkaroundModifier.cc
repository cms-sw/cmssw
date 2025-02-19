/*
 * ===========================================================================
 *
 *       Filename:  RecoTauTagInfoWorkaroundModifer
 *
 *    Description:  Add the PFTauTagInfoRef back to PFTaus so things don't
 *                  break.
 *
 *         Author:  Evan K. Friis (UC Davis)
 *
 * ===========================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

namespace reco { namespace tau {

class RecoTauTagInfoWorkaroundModifer : public RecoTauModifierPlugin {
  public:
    explicit RecoTauTagInfoWorkaroundModifer(const edm::ParameterSet &pset);
    virtual ~RecoTauTagInfoWorkaroundModifer() {}
    void operator()(PFTau&) const;
    // Called by base class
    void beginEvent();
  private:
    edm::InputTag pfTauTagInfoSrc_;
    edm::Handle<PFTauTagInfoCollection> infos_;
};

RecoTauTagInfoWorkaroundModifer::RecoTauTagInfoWorkaroundModifer(
    const edm::ParameterSet &pset):RecoTauModifierPlugin(pset) {
  pfTauTagInfoSrc_ = pset.getParameter<edm::InputTag>("pfTauTagInfoSrc");
}

// Load our tau tag infos from the event
void RecoTauTagInfoWorkaroundModifer::beginEvent() {
  evt()->getByLabel(pfTauTagInfoSrc_, infos_);
}

void RecoTauTagInfoWorkaroundModifer::operator()(PFTau& tau) const {
  // Find the PFTauTagInfo that comes from the same PFJet
  PFJetRef tauJetRef = tau.jetRef();
  for(size_t iInfo = 0; iInfo < infos_->size(); ++iInfo) {
    // Get jet ref from tau tag info
    PFTauTagInfoRef infoRef = PFTauTagInfoRef(infos_, iInfo);
    PFJetRef infoJetRef = infoRef->pfjetRef();
    // Check if they come from the same jet
    if (infoJetRef == tauJetRef) {
      // This tau "comes" from this PFJetRef
      tau.setpfTauTagInfoRef(infoRef);
      break;
    }
  }
}
}}  // end namespace reco::tau
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauTagInfoWorkaroundModifer,
    "RecoTauTagInfoWorkaroundModifer");
