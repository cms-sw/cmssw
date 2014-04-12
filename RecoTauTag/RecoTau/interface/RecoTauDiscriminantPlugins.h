#ifndef RecoTauTag_RecoTau_RecoTauDiscriminantPlugins_h
#define RecoTauTag_RecoTau_RecoTauDiscriminantPlugins_h

/*
 * Common base classes for the plugins used in the
 * that produce PFTau discriminants.  A PFTauDiscriminant
 * (not discriminator) is a vector of doubles associated
 * to a given tau, and represent some observable(s) for that
 * tau.
 *
 * Author: Evan K. Friis, UC Davis
 *
 */

#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include <vector>

namespace reco { namespace tau {

// Convert a MVA name (i.e. Pt, Eta) to the appropriate plugin name.
//  Example: discPluginName("Pt") -> "RecoTauDiscriminationPt"
inline std::string discPluginName(const std::string& mvaName) {
  return "RecoTauDiscrimination" + mvaName;
}

class RecoTauDiscriminantPlugin : public RecoTauEventHolderPlugin {
  public:
    explicit RecoTauDiscriminantPlugin(const edm::ParameterSet& pset):
      RecoTauEventHolderPlugin(pset){}
    virtual ~RecoTauDiscriminantPlugin() {}
    virtual void beginEvent() {}
    // Get an observable
    virtual std::vector<double> operator()(const reco::PFTauRef& pfTau) const=0;
};

// Build a discriminant using a unary PFTau function that returns a single value
template<double Function(const reco::PFTau&)>
  class RecoTauDiscriminantFunctionPlugin : public RecoTauDiscriminantPlugin {
  public:
    explicit RecoTauDiscriminantFunctionPlugin(const edm::ParameterSet& pset):
      RecoTauDiscriminantPlugin(pset){}

    virtual ~RecoTauDiscriminantFunctionPlugin(){}

    virtual std::vector<double> operator()(const reco::PFTauRef& pfTau) const {
      std::vector<double> output(1, Function(*pfTau));
      return output;
    }
};

// Build a discriminant using a unary PFTau function that returns a vector of values
template<std::vector<double> Function(const reco::PFTau&)>
  class RecoTauDiscriminantVectorFunctionPlugin :
    public RecoTauDiscriminantPlugin {
  public:
    explicit RecoTauDiscriminantVectorFunctionPlugin(const edm::ParameterSet& pset):
      RecoTauDiscriminantPlugin(pset){}

    virtual ~RecoTauDiscriminantVectorFunctionPlugin() {}

    virtual std::vector<double> operator()(const reco::PFTauRef& pfTau) const {
      return Function(*pfTau);
    }
};
} } // end reco::tau

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<reco::tau::RecoTauDiscriminantPlugin* (const edm::ParameterSet&)> RecoTauDiscriminantPluginFactory;
#endif
