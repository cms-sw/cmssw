/*
 * ===========================================================================
 *
 *       Filename:  RecoTauObjectEmbedder
 *
 *    Description:  Embed truth information into (currently unused)
 *                  tau variables.  This is a hack to allow some PAT-like
 *                  operations on taus without introducing new dependencies.
 *
 *         Author:  Evan K. Friis (UC Davis)
 *
 * ===========================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

namespace reco { namespace tau {

namespace helpers {
unsigned int nCharged(const GenJet& jet) {
  unsigned int output = 0;
  for(auto const& cand : jet.getJetConstituents()) {
    if (cand->charge())
      ++output;
  }
  return output;
}

unsigned int nGammas(const GenJet& jet) {
  unsigned int output = 0;
  for(auto const& cand : jet.getJetConstituents()) {
    if (cand->pdgId()==22)
      ++output;
  }
  return output;
}

unsigned int nCharged(const PFTau& tau) {
  return tau.signalPFChargedHadrCands().size();
}
unsigned int nGammas(const PFTau& tau) {
  return tau.signalPiZeroCandidates().size();
}
}

template<typename T>
class RecoTauObjectEmbedder : public RecoTauModifierPlugin {
  public:
  explicit RecoTauObjectEmbedder(const edm::ParameterSet &pset, edm::ConsumesCollector &&iC)
    :RecoTauModifierPlugin(pset,std::move(iC)),
        jetMatchSrc_(pset.getParameter<edm::InputTag>("jetTruthMatch")) {}
    ~RecoTauObjectEmbedder() override {}
    void operator()(PFTau&) const override;
    void beginEvent() override;
  private:
    edm::InputTag jetMatchSrc_;
    edm::Handle<edm::Association<T> > jetMatch_;
};

// Update our handle to the matching
template<typename T>
void RecoTauObjectEmbedder<T>::beginEvent() {
  evt()->getByLabel(jetMatchSrc_, jetMatch_);
}

template<typename T>
void RecoTauObjectEmbedder<T>::operator()(PFTau& tau) const {
  // Get the matched object that is matched to the same jet as the current tau,
  // if it exists
  edm::Ref<T> matchedObject = (*jetMatch_)[tau.jetRef()];
  if (matchedObject.isNonnull()) {
    // Store our matched object information
    tau.setalternatLorentzVect(matchedObject->p4());
    // Store our generator decay mode
    tau.setbremsRecoveryEOverPLead(
        reco::tau::translateDecayMode(
            helpers::nCharged(*matchedObject),
            helpers::nGammas(*matchedObject)/2)
        );
  } else {
    tau.setbremsRecoveryEOverPLead(-10);
  }
}
}}  // end namespace reco::tau

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauObjectEmbedder<reco::GenJetCollection>,
    "RecoTauTruthEmbedder");

DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauObjectEmbedder<reco::PFTauCollection>,
    "RecoTauPFTauEmbedder");
