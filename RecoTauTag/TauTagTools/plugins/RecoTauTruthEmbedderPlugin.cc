/*
 * ===========================================================================
 *
 *       Filename:  RecoTauTruthEmbedder
 *
 *    Description:  Embed truth information into (currently unused)
 *                  tau variables.  This is a hack to allow some PAT-like
 *                  operations on taus without introducing new dependencies.
 *
 *         Author:  Evan K. Friis (UC Davis)
 *
 * ===========================================================================
 */

#include <boost/foreach.hpp>

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/PFTauDecayModeTools.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Association.h"

namespace reco { namespace tau {

namespace helpers {
unsigned int nCharged(const GenJet& jet) {
  unsigned int output = 0;
  BOOST_FOREACH(const CandidatePtr &cand, jet.getJetConstituents()) {
    if (cand->charge())
      ++output;
  }
  return output;
}

unsigned int nGammas(const GenJet& jet) {
  unsigned int output = 0;
  BOOST_FOREACH(const CandidatePtr &cand, jet.getJetConstituents()) {
    if (cand->pdgId()==22)
      ++output;
  }
  return output;
}
}

class RecoTauTruthEmbedder : public RecoTauModifierPlugin {
  public:
    explicit RecoTauTruthEmbedder(const edm::ParameterSet &pset)
        :RecoTauModifierPlugin(pset),
        jetMatchSrc_(pset.getParameter<edm::InputTag>("jetTruthMatch")) {}
    virtual ~RecoTauTruthEmbedder() {}
    virtual void operator()(PFTau&) const;
    virtual void beginEvent();
  private:
    edm::InputTag jetMatchSrc_;
    edm::Handle<edm::Association<GenJetCollection> > jetMatch_;
};

// Update our handle to the matching
void RecoTauTruthEmbedder::beginEvent() {
  evt()->getByLabel(jetMatchSrc_, jetMatch_);
}

void RecoTauTruthEmbedder::operator()(PFTau& tau) const {
  // Get the matched truth tau if it exists
  GenJetRef truth = (*jetMatch_)[tau.jetRef()];
  if (truth.isNonnull()) {
    // Store our generator level information
    tau.setalternatLorentzVect(truth->p4());
    // Store our generator decay mode
    tau.setbremsRecoveryEOverPLead(
        reco::tau::translateDecayMode(
            helpers::nCharged(*truth),
            helpers::nGammas(*truth)/2)
        );
  } else {
    tau.setbremsRecoveryEOverPLead(-10);
  }
}
}}  // end namespace reco::tau
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauTruthEmbedder,
    "RecoTauTruthEmbedder");
