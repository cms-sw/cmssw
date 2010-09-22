#ifndef RecoTauTag_RecoTau_RecoTauPiZeroPlugins_h
#define RecoTauTag_RecoTau_RecoTauPiZeroPlugins_h

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"
#include <vector>

namespace reco {

  class PFJet;
  class RecoTauPiZero;

  namespace tau {

    class RecoTauPiZeroBuilderPlugin : public RecoTauNamedPlugin
    {
      public:
        explicit RecoTauPiZeroBuilderPlugin(const edm::ParameterSet& pset):
          RecoTauNamedPlugin(pset) {}
        virtual ~RecoTauPiZeroBuilderPlugin() {}
        /// Build a collection of piZeros from objects in the input jet
        virtual std::vector<RecoTauPiZero> operator()(const PFJet&) const = 0;
    };

    class RecoTauPiZeroQualityPlugin : public RecoTauNamedPlugin
    {
      public:
        explicit RecoTauPiZeroQualityPlugin(const edm::ParameterSet& pset):
          RecoTauNamedPlugin(pset) {}
        virtual ~RecoTauPiZeroQualityPlugin() {}
        /// Return a number indicating the quality of this PiZero
        virtual double operator()(const RecoTauPiZero&) const = 0;
    };
  }

}

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<reco::tau::RecoTauPiZeroQualityPlugin* (const edm::ParameterSet&)> RecoTauPiZeroQualityPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauPiZeroBuilderPlugin* (const edm::ParameterSet&)> RecoTauPiZeroBuilderPluginFactory;


#endif


