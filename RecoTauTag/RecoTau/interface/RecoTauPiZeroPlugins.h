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
#include <boost/ptr_container/ptr_vector.hpp>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"

namespace reco {
// Forward declarations
class PFJet;
class RecoTauPiZero;
namespace tau {

class RecoTauPiZeroBuilderPlugin : public RecoTauEventHolderPlugin {
  public:
    // Return a vector of pointers
    typedef boost::ptr_vector<RecoTauPiZero> PiZeroVector;
    // Storing the result in an auto ptr on function return allows
    // allows us to safely release the ptr_vector in the virtual function
    typedef std::auto_ptr<PiZeroVector> return_type;
    explicit RecoTauPiZeroBuilderPlugin(const edm::ParameterSet& pset):
        RecoTauEventHolderPlugin(pset) {}
    virtual ~RecoTauPiZeroBuilderPlugin() {}
    /// Build a collection of piZeros from objects in the input jet
    virtual return_type operator()(const PFJet&) const = 0;
    /// Hook called at the beginning of the event.
    virtual void beginEvent() {};
};

class RecoTauPiZeroQualityPlugin : public RecoTauNamedPlugin {
  public:
    explicit RecoTauPiZeroQualityPlugin(const edm::ParameterSet& pset):
        RecoTauNamedPlugin(pset) {}
    virtual ~RecoTauPiZeroQualityPlugin() {}
    /// Return a number indicating the quality of this PiZero
    virtual double operator()(const RecoTauPiZero&) const = 0;
};
}}  // end namespace reco::tau

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<reco::tau::RecoTauPiZeroQualityPlugin*
(const edm::ParameterSet&)> RecoTauPiZeroQualityPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauPiZeroBuilderPlugin*
(const edm::ParameterSet&)> RecoTauPiZeroBuilderPluginFactory;
#endif
