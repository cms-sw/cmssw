#ifndef RecoTauTag_RecoTau_PFRecoTauChargedHadronPlugins_h
#define RecoTauTag_RecoTau_PFRecoTauChargedHadronPlugins_h

/*
 * PFRecoTauChargedHadronPlugins
 *
 * Author: Christian Veelken, LLR
 *
 * Base classes for plugins that construct and rank PFRecoTauChargedHadron
 * objects from a jet.  The builder plugin has an abstract function
 * that takes a PFJet and returns a list of reconstructed photons in
 * the jet.
 *
 * The quality plugin has an abstract function that takes a reference
 * to a PFRecoTauChargedHadron and returns a double indicating the quality of
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
class PFRecoTauChargedHadron;

namespace tau {

class PFRecoTauChargedHadronBuilderPlugin : public RecoTauEventHolderPlugin 
{
 public:
  // Return a vector of pointers
  typedef boost::ptr_vector<PFRecoTauChargedHadron> ChargedHadronVector;
  // Storing the result in an auto ptr on function return allows
  // allows us to safely release the ptr_vector in the virtual function
  typedef std::auto_ptr<ChargedHadronVector> return_type;
  explicit PFRecoTauChargedHadronBuilderPlugin(const edm::ParameterSet& pset)
    : RecoTauEventHolderPlugin(pset) 
  {}
  virtual ~PFRecoTauChargedHadronBuilderPlugin() {}
  /// Build a collection of chargedHadrons from objects in the input jet
  virtual return_type operator()(const PFJet&) const = 0;
  /// Hook called at the beginning of the event.
  virtual void beginEvent() {}
};

class PFRecoTauChargedHadronQualityPlugin : public RecoTauNamedPlugin 
{
 public:
  explicit PFRecoTauChargedHadronQualityPlugin(const edm::ParameterSet& pset)
    : RecoTauNamedPlugin(pset) 
  {}
  virtual ~PFRecoTauChargedHadronQualityPlugin() {}
  /// Return a number indicating the quality of this chargedHadron
  virtual double operator()(const PFRecoTauChargedHadron&) const = 0;
};

}} // end namespace reco::tau

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<reco::tau::PFRecoTauChargedHadronQualityPlugin*(const edm::ParameterSet&)> PFRecoTauChargedHadronQualityPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::PFRecoTauChargedHadronBuilderPlugin*(const edm::ParameterSet&)> PFRecoTauChargedHadronBuilderPluginFactory;

#endif
