/*
 * RecoTauGenericSoftTwoProngTausCleanerPlugin
 *
 * Author: Christian Veelken, NICPB Tallinn
 *
 * Remove 2-prong PFTaus with a low pT track, in order to reduce rate of 1-prong taus migrating to 2-prong decay mode
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"

namespace reco { namespace tau {

template<class TauType>
class RecoTauGenericSoftTwoProngTausCleanerPlugin : public RecoTauCleanerPlugin<TauType>
{
 public:
  RecoTauGenericSoftTwoProngTausCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);

  // Get ranking value for a given tau Ref
  double operator()(const edm::Ref<std::vector<TauType> >&) const override;
 private:
  double minTrackPt_;
};

template<class TauType>
RecoTauGenericSoftTwoProngTausCleanerPlugin<TauType>::RecoTauGenericSoftTwoProngTausCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
  : RecoTauCleanerPlugin<TauType>(pset,std::move(iC)) 
{
  minTrackPt_ = pset.getParameter<double>("minTrackPt");
}

namespace {
  const reco::Track* getTrackFromChargedHadron(const reco::PFRecoTauChargedHadron& chargedHadron) {
    // Charged hadron made from track (reco::Track) - RECO/AOD only
    if ( chargedHadron.getTrack().isNonnull()) {
      return chargedHadron.getTrack().get();
    }
    const pat::PackedCandidate* chargedPFPCand = dynamic_cast<const pat::PackedCandidate*> (&*chargedHadron.getChargedPFCandidate());
    if (chargedPFPCand) {
        if (chargedPFPCand->hasTrackDetails())
          return &chargedPFPCand->pseudoTrack();
    }
    return nullptr;
  }
}

template<class TauType>
double RecoTauGenericSoftTwoProngTausCleanerPlugin<TauType>::operator()(const edm::Ref<std::vector<TauType> >& tau) const 
{
  double result = 0.;
  const std::vector<PFRecoTauChargedHadron>& chargedHadrons = tau->signalTauChargedHadronCandidates();
  if ( chargedHadrons.size() == 2 ) {
    for ( std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
	  chargedHadron != chargedHadrons.end(); ++chargedHadron ) {
      const reco::Track* track = getTrackFromChargedHadron(*chargedHadron);
      if ( !(track && track->pt() > minTrackPt_) ) result += 1.e+3;
    }
  }
  return result;
}

template class RecoTauGenericSoftTwoProngTausCleanerPlugin<reco::PFTau>;
typedef RecoTauGenericSoftTwoProngTausCleanerPlugin<reco::PFTau> RecoTauSoftTwoProngTausCleanerPlugin;
template class RecoTauGenericSoftTwoProngTausCleanerPlugin<reco::PFBaseTau>;
typedef RecoTauGenericSoftTwoProngTausCleanerPlugin<reco::PFBaseTau> RecoBaseTauSoftTwoProngTausCleanerPlugin;


}} // end namespace reco::tau

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, 
    reco::tau::RecoTauSoftTwoProngTausCleanerPlugin, 
    "RecoTauSoftTwoProngTausCleanerPlugin");

DEFINE_EDM_PLUGIN(RecoBaseTauCleanerPluginFactory, 
    reco::tau::RecoBaseTauSoftTwoProngTausCleanerPlugin, 
    "RecoBaseTauSoftTwoProngTausCleanerPlugin");