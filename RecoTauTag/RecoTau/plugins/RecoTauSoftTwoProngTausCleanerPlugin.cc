/*
 * RecoTauSoftTwoProngTausCleanerPlugin
 *
 * Author: Christian Veelken, NICPB Tallinn
 *
 * Remove 2-prong PFTaus with a low pT track, in order to reduce rate of 1-prong taus migrating to 2-prong decay mode
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

namespace reco { namespace tau {

class RecoTauSoftTwoProngTausCleanerPlugin : public RecoTauCleanerPlugin 
{
 public:
  RecoTauSoftTwoProngTausCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);

  // Get ranking value for a given tau Ref
  double operator()(const reco::PFTauRef&) const override;
 private:
  double minTrackPt_;
};

RecoTauSoftTwoProngTausCleanerPlugin::RecoTauSoftTwoProngTausCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
  : RecoTauCleanerPlugin(pset,std::move(iC)) 
{
  minTrackPt_ = pset.getParameter<double>("minTrackPt");
}

double RecoTauSoftTwoProngTausCleanerPlugin::operator()(const reco::PFTauRef& tau) const 
{
  double result = 0.;
  const std::vector<PFRecoTauChargedHadron>& chargedHadrons = tau->signalTauChargedHadronCandidates();
  if ( chargedHadrons.size() == 2 ) {
    for ( std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = chargedHadrons.begin();
	  chargedHadron != chargedHadrons.end(); ++chargedHadron ) {
      if ( !(chargedHadron->getTrack().get() && chargedHadron->getTrack()->pt() > minTrackPt_) ) result += 1.e+3;
    }
  }
  return result;
}

}} // end namespace reco::tau

// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, 
    reco::tau::RecoTauSoftTwoProngTausCleanerPlugin, 
    "RecoTauSoftTwoProngTausCleanerPlugin");
