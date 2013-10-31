#ifndef RecoTauTag_RecoTau_RecoTauBuilderPlugin_h
#define RecoTauTag_RecoTau_RecoTauBuilderPlugin_h

/*
 * RecoTauBuilderPlugins
 *
 * Author: Evan K. Friis (UC Davis)
 *
 * Classes for building new and modifying existing PFTaus from PFJets and
 * reconstructed PiZeros.
 *
 * RecoTauBuilderPlugin is the base class for any algorithm that constructs
 * taus.  Algorithms should override the abstract function
 *
 * std::vector<PFTau> operator()(const PFJet&, const
 * std::vector<RecoTauPiZero>&) const;
 *
 * implementing it such that a list of taus a produced for a given jet and its
 * associated collection of PiZeros.
 *
 * RecoTauModifierPlugin takes an input tau and modifies it.
 *
 * Both plugins inherit from RecoTauEventHolderPlugin, which provides the
 * methods
 *
 *    const edm::Event* evt() const; const edm::EventSetup* evtSetup()
 *
 * to retrieve the current event if necessary.
 *
 * $Id $
 *
 */

#include <boost/ptr_container/ptr_vector.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>

namespace reco { namespace tau {

/* Class that constructs PFTau(s) from a PFJet and its associated PiZeros */
class RecoTauBuilderPlugin : public RecoTauEventHolderPlugin 
{
 public:
  typedef boost::ptr_vector<reco::PFTau> output_type;
  typedef std::auto_ptr<output_type> return_type;

  explicit RecoTauBuilderPlugin(const edm::ParameterSet& pset)
    : RecoTauEventHolderPlugin(pset),
      // The vertex association configuration is specified with the quality cuts.
      vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts")) 
  {
    pfCandSrc_ = pset.getParameter<edm::InputTag>("pfCandSrc");
  };

  virtual ~RecoTauBuilderPlugin() {}

  /// Construct one or more PFTaus from the a PFJet and its asscociated
  /// reconstructed PiZeros and regional extras i.e. objects in a 0.8 cone
  /// about the jet
  virtual return_type operator()(
	    const reco::PFJetRef&, const 
	    std::vector<reco::PFRecoTauChargedHadron>&, 
	    const std::vector<reco::RecoTauPiZero>&, 
	    const std::vector<PFCandidatePtr>&) const = 0;

  /// Hack to be able to convert Ptrs to Refs
  const edm::Handle<PFCandidateCollection>& getPFCands() const { return pfCands_; };

  /// Get primary vertex associated to this jet
  reco::VertexRef primaryVertex(const reco::PFJetRef& jet) const { return vertexAssociator_.associatedVertex(*jet); }

  // Hook called by base class at the beginning of each event. Used to update
  // handle to PFCandidates
  virtual void beginEvent();
    
 private:
  edm::InputTag pfCandSrc_;
  // Handle to PFCandidates needed to build Refs
  edm::Handle<PFCandidateCollection> pfCands_;
  reco::tau::RecoTauVertexAssociator vertexAssociator_;
};

/* Class that updates a PFTau's members (i.e. electron variables) */
class RecoTauModifierPlugin : public RecoTauEventHolderPlugin 
{
 public:
  explicit RecoTauModifierPlugin(const edm::ParameterSet& pset)
    : RecoTauEventHolderPlugin(pset)
  {}
  virtual ~RecoTauModifierPlugin() {}
  // Modify an existing PFTau (i.e. add electron rejection, etc)
  virtual void operator()(PFTau&) const = 0;
  virtual void beginJob(edm::EDProducer*) {}
  virtual void beginEvent() {}
  virtual void endEvent() {}
};

/* Class that returns a double value indicating the quality of a given tau */
class RecoTauCleanerPlugin : public RecoTauEventHolderPlugin 
{
 public:
  explicit RecoTauCleanerPlugin(const edm::ParameterSet& pset)
    : RecoTauEventHolderPlugin(pset)
  {}
  virtual ~RecoTauCleanerPlugin() {}
  // Modify an existing PFTau (i.e. add electron rejection, etc)
  virtual double operator()(const PFTauRef&) const = 0;
  virtual void beginEvent() {}
};
} } // end namespace reco::tau

#include "FWCore/PluginManager/interface/PluginFactory.h"

typedef edmplugin::PluginFactory<reco::tau::RecoTauBuilderPlugin*(const edm::ParameterSet&)> RecoTauBuilderPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauModifierPlugin*(const edm::ParameterSet&)> RecoTauModifierPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauCleanerPlugin*(const edm::ParameterSet&)> RecoTauCleanerPluginFactory;

#endif
