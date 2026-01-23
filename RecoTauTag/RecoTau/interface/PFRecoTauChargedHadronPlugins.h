#ifndef RecoTauTag_RecoTau_PFRecoTauChargedHadronPlugins_h
#define RecoTauTag_RecoTau_PFRecoTauChargedHadronPlugins_h

/*
 * PFRecoTauChargedHadronPlugins
 *
 * Author: Christian Veelken, LLR
 *
 * Base classes for plugins that construct and rank PFRecoTauChargedHadron
 * objects from a jet.  The builder plugin has an abstract function
 * that takes a Jet and returns a list of reconstructed photons in
 * the jet.
 *
 * The quality plugin has an abstract function that takes a reference
 * to a PFRecoTauChargedHadron and returns a double indicating the quality of
 * the candidate.  Lower numbers are better.
 *
 */

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/JetFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"

#include <vector>

namespace reco {

  // Forward declarations
  class PFRecoTauChargedHadron;

  namespace tau {

    class PFRecoTauChargedHadronBuilderPlugin : public RecoTauEventHolderPlugin {
    public:
      // Return a vector of pointers
      typedef std::vector<std::unique_ptr<PFRecoTauChargedHadron>> ChargedHadronVector;
      typedef ChargedHadronVector return_type;
      explicit PFRecoTauChargedHadronBuilderPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
          : RecoTauEventHolderPlugin(pset) {}
      ~PFRecoTauChargedHadronBuilderPlugin() override {}
      /// Build a collection of chargedHadrons from objects in the input jet
      virtual return_type operator()(const Jet&) const = 0;
      /// Hook called at the beginning of the event.
      void beginEvent() override {}
    };

    class PFRecoTauChargedHadronQualityPlugin : public RecoTauNamedPlugin {
    public:
      explicit PFRecoTauChargedHadronQualityPlugin(const edm::ParameterSet& pset) : RecoTauNamedPlugin(pset) {}
      ~PFRecoTauChargedHadronQualityPlugin() override {}
      /// Return a number indicating the quality of this chargedHadron
      virtual double operator()(const PFRecoTauChargedHadron&) const = 0;
    };

  }  // namespace tau
}  // namespace reco

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<reco::tau::PFRecoTauChargedHadronQualityPlugin*(const edm::ParameterSet&)>
    PFRecoTauChargedHadronQualityPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::PFRecoTauChargedHadronBuilderPlugin*(const edm::ParameterSet&,
                                                                                 edm::ConsumesCollector&& iC)>
    PFRecoTauChargedHadronBuilderPluginFactory;

#endif
