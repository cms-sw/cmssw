#ifndef RecoTauTag_RecoTau_RecoTauPiZeroPlugins_h
#define RecoTauTag_RecoTau_RecoTauPiZeroPlugins_h

/*
 * RecoTauPiZeroPlugins
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Base classes for plugins that construct and rank RecoTauPiZero
 * objects from a jet.  The builder plugin has an abstract function
 * that takes a PFJet and returns a list of reconstructed photons in
 * the jet.
 *
 * The quality plugin has an abstract function that takes a reference
 * to a RecoTauPiZero and returns a double indicating the quality of
 * the candidate.  Lower numbers are better.
 *
 */

#include <vector>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/JetReco/interface/JetFwd.h"
#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"
#include "DataFormats/TauReco/interface/RecoTauPiZeroFwd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace reco {
  namespace tau {

    class RecoTauPiZeroBuilderPlugin : public RecoTauEventHolderPlugin {
    public:
      // Return a vector of pointers
      typedef std::vector<std::unique_ptr<RecoTauPiZero>> PiZeroVector;
      typedef PiZeroVector return_type;
      explicit RecoTauPiZeroBuilderPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
          : RecoTauEventHolderPlugin(pset) {}
      ~RecoTauPiZeroBuilderPlugin() override {}
      /// Build a collection of piZeros from objects in the input jet
      virtual return_type operator()(const Jet&) const = 0;
      /// Hook called at the beginning of the event.
      void beginEvent() override {}
    };

    class RecoTauPiZeroQualityPlugin : public RecoTauNamedPlugin {
    public:
      explicit RecoTauPiZeroQualityPlugin(const edm::ParameterSet& pset) : RecoTauNamedPlugin(pset) {}
      ~RecoTauPiZeroQualityPlugin() override {}
      /// Return a number indicating the quality of this PiZero
      virtual double operator()(const RecoTauPiZero&) const = 0;
    };
  }  // namespace tau
}  // namespace reco

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<reco::tau::RecoTauPiZeroQualityPlugin*(const edm::ParameterSet&)>
    RecoTauPiZeroQualityPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauPiZeroBuilderPlugin*(const edm::ParameterSet&,
                                                                        edm::ConsumesCollector&& iC)>
    RecoTauPiZeroBuilderPluginFactory;
#endif
