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
 *
 */

#include <boost/ptr_container/ptr_vector.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "RecoTauTag/RecoTau/interface/RecoTauPluginsCommon.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Common/interface/Handle.h"

#include <vector>

namespace reco { namespace tau {

/* Class that constructs PFTau(s) from a Jet and its associated PiZeros */
template<class TauType, class PFType>
class RecoTauBuilderPlugin : public RecoTauEventHolderPlugin 
{
 public:
  typedef boost::ptr_vector<TauType> output_type;
  typedef std::auto_ptr<output_type> return_type;
  typedef std::vector<PFType> PFCandidateCollection;

  explicit RecoTauBuilderPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector && iC)
    : RecoTauEventHolderPlugin(pset),
      // The vertex association configuration is specified with the quality cuts.
    vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"),std::move(iC))
  {
    pfCandSrc_ = pset.getParameter<edm::InputTag>("pfCandSrc");
    pfCand_token = iC.consumes<PFCandidateCollection>(pfCandSrc_);
  };

  ~RecoTauBuilderPlugin() override {}

  /// Construct one or more PFTaus from the a PFJet and its asscociated
  /// reconstructed PiZeros and regional extras i.e. objects in a 0.8 cone
  /// about the jet
  virtual return_type operator()(
	    const reco::JetBaseRef&, const 
	    std::vector<reco::PFRecoTauChargedHadron>&, 
	    const std::vector<reco::RecoTauPiZero>&, 
	    const std::vector<CandidatePtr>&) const = 0;

  /// Hack to be able to convert Ptrs to Refs
  const edm::Handle<PFCandidateCollection>& getPFCands() const { return pfCands_; };

  /// Get primary vertex associated to this jet
  reco::VertexRef primaryVertex(const reco::JetBaseRef& jet) const { return vertexAssociator_.associatedVertex(*jet); }
  /// Get primary vertex associated to this tau
  reco::VertexRef primaryVertex(const TauType& tau, bool useJet=false) const { return vertexAssociator_.associatedVertex(tau, useJet); }

  // Hook called by base class at the beginning of each event. Used to update
  // handle to PFCandidates
  void beginEvent() override;
    
 private:
  edm::InputTag pfCandSrc_;
  // Handle to PFCandidates needed to build Refs
  edm::Handle<PFCandidateCollection> pfCands_;
  reco::tau::RecoTauVertexAssociator vertexAssociator_;
  edm::EDGetTokenT<PFCandidateCollection> pfCand_token;
};

/* Class that updates a PFTau's members (i.e. electron variables) */
template<class TauType>
class RecoTauModifierPlugin : public RecoTauEventHolderPlugin 
{
 public:
  explicit RecoTauModifierPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
    : RecoTauEventHolderPlugin(pset)
  {}
  ~RecoTauModifierPlugin() override {}
  // Modify an existing PFTau (i.e. add electron rejection, etc)
  virtual void operator()(TauType&) const = 0;
  void beginEvent() override {}
  virtual void endEvent() {}
};

/* Class that returns a double value indicating the quality of a given tau */
template<class TauType>
class RecoTauCleanerPlugin : public RecoTauEventHolderPlugin 
{
 public:
  explicit RecoTauCleanerPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector && iC )
    : RecoTauEventHolderPlugin(pset)
  {}
  ~RecoTauCleanerPlugin() override {}
  // Modify an existing PFTau (i.e. add electron rejection, etc)
  virtual double operator()(const edm::Ref<std::vector<TauType> >&) const = 0;
  void beginEvent() override {}
};

// Update our reference to the PFCandidates & PVs
template<class TauType, class PFType>
void RecoTauBuilderPlugin<TauType, PFType>::beginEvent() {
  vertexAssociator_.setEvent(*evt());
  evt()->getByToken(pfCand_token, pfCands_);
}

} } // end namespace reco::tau

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

typedef edmplugin::PluginFactory<reco::tau::RecoTauBuilderPlugin<reco::PFTau, reco::PFCandidate>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)> RecoTauBuilderPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauModifierPlugin<reco::PFTau>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)> RecoTauModifierPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauCleanerPlugin<reco::PFTau>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)> RecoTauCleanerPluginFactory;

typedef edmplugin::PluginFactory<reco::tau::RecoTauBuilderPlugin<reco::PFBaseTau, pat::PackedCandidate>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)> RecoBaseTauBuilderPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauModifierPlugin<reco::PFBaseTau>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)> RecoBaseTauModifierPluginFactory;
typedef edmplugin::PluginFactory<reco::tau::RecoTauCleanerPlugin<reco::PFBaseTau>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)> RecoBaseTauCleanerPluginFactory;

#endif
