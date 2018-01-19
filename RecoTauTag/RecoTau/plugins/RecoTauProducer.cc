/*
 * RecoGenericTauProducer
 *
 * Interface between the various tau algorithms and the edm::Event.  The
 * RecoGenericTauProducer takes as data input is a collection (view) of reco::PFJets,
 * and Jet-PiZero assoications that give the reco::RecoTauPiZeros for those
 * jets.  The actual building of taus is done by the list of builders - each of
 * which constructs a PFTau for each PFJet.  The output collection may have
 * multiple taus for each PFJet - these overlaps are to be resolved by the
 * RecoTauCleaner module.
 *
 * Additionally, there are "modifier" plugins, which can do things like add the
 * lead track significance, or electron rejection variables.
 *
 * Authors: Evan K. Friis (UC Davis),
 *          Christian Veelken (LLR)
 *
 */
#include "boost/bind.hpp"
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/foreach.hpp>

#include <algorithm>
#include <functional>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFJetChargedHadronAssociation.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/Association.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

template<class TauType, class PFType>
class RecoGenericTauProducer : public edm::stream::EDProducer<> 
{
 public:
  typedef typename reco::tau::RecoTauBuilderPlugin<TauType, PFType> Builder;
  typedef typename reco::tau::RecoTauModifierPlugin<TauType> Modifier;
  typedef boost::ptr_vector<Builder> BuilderList;
  typedef boost::ptr_vector<Modifier> ModifierList;

  explicit RecoGenericTauProducer(const edm::ParameterSet& pset);
  ~RecoGenericTauProducer() override {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  TauType buildNullTau(const edm::RefToBase<reco::Jet>& jetRef);
  void setJetRefs(typename reco::tau::RecoTauBuilderPlugin<TauType, PFType>::output_type taus, const edm::RefToBase<reco::Jet>& jetRef);

 private:
  edm::InputTag jetSrc_;
  edm::InputTag jetRegionSrc_;
  edm::InputTag chargedHadronSrc_;
  edm::InputTag piZeroSrc_;

  double minJetPt_;
  double maxJetAbsEta_;
 //token definition
  edm::EDGetTokenT<reco::JetView> jet_token;
  edm::EDGetTokenT<edm::Association<typename TauType::TauJetCollection> > jetRegion_token;
  edm::EDGetTokenT<reco::PFJetChargedHadronAssociation> chargedHadron_token;
  edm::EDGetTokenT<reco::JetPiZeroAssociation> piZero_token;

  BuilderList builders_;
  ModifierList modifiers_;
  // Optional selection on the output of the taus
  std::auto_ptr<StringCutObjectSelector<TauType> > outputSelector_;
  // Whether or not to add build a tau from a jet for which the builders
  // return no taus.  The tau will have no content, only the four vector of
  // the orginal jet.
  bool buildNullTaus_;
};

template<class TauType, class PFType>
RecoGenericTauProducer<TauType, PFType>::RecoGenericTauProducer(const edm::ParameterSet& pset) 
{
  jetSrc_ = pset.getParameter<edm::InputTag>("jetSrc");
  jetRegionSrc_ = pset.getParameter<edm::InputTag>("jetRegionSrc");
  chargedHadronSrc_ = pset.getParameter<edm::InputTag>("chargedHadronSrc");
  piZeroSrc_ = pset.getParameter<edm::InputTag>("piZeroSrc");
  
  minJetPt_ = ( pset.exists("minJetPt") ) ? pset.getParameter<double>("minJetPt") : -1.0;
  maxJetAbsEta_ = ( pset.exists("maxJetAbsEta") ) ? pset.getParameter<double>("maxJetAbsEta") : 99.0;
  //consumes definition
  jet_token=consumes<reco::JetView>(jetSrc_);
  jetRegion_token = consumes<edm::Association<typename TauType::TauJetCollection> >(jetRegionSrc_);
  chargedHadron_token = consumes<reco::PFJetChargedHadronAssociation>(chargedHadronSrc_); 
  piZero_token = consumes<reco::JetPiZeroAssociation>(piZeroSrc_);

  typedef std::vector<edm::ParameterSet> VPSet;
  // Get each of our tau builders
  const VPSet& builders = pset.getParameter<VPSet>("builders");
  for ( VPSet::const_iterator builderPSet = builders.begin();
	builderPSet != builders.end(); ++builderPSet ) {
    // Get plugin name
    const std::string& pluginType = builderPSet->getParameter<std::string>("plugin");
    // Build the plugin
    builders_.push_back(edmplugin::PluginFactory<reco::tau::RecoTauBuilderPlugin<TauType, PFType>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)>::get()->create(pluginType, *builderPSet, consumesCollector()));
  }

  const VPSet& modfiers = pset.getParameter<VPSet>("modifiers");
  for ( VPSet::const_iterator modfierPSet = modfiers.begin();
	modfierPSet != modfiers.end(); ++modfierPSet) {
    // Get plugin name
    const std::string& pluginType = modfierPSet->getParameter<std::string>("plugin");
    // Build the plugin
    reco::tau::RecoTauModifierPlugin<TauType>* plugin = nullptr;
    plugin = edmplugin::PluginFactory<reco::tau::RecoTauModifierPlugin<TauType>*(const edm::ParameterSet&, edm::ConsumesCollector &&iC)>::get()->create(pluginType, *modfierPSet, consumesCollector());
    modifiers_.push_back(plugin);
  }

  // Check if we want to apply a final output selection
  if ( pset.exists("outputSelection") ) {
    std::string selection = pset.getParameter<std::string>("outputSelection");
    if ( selection != "" ) {
      outputSelector_.reset(new StringCutObjectSelector<TauType>(selection));
    }
  }
  buildNullTaus_ = pset.getParameter<bool>("buildNullTaus");

  produces<std::vector<TauType> >();
}

template<class TauType, class PFType>
void RecoGenericTauProducer<TauType, PFType>::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  // Get the jet input collection via a view of Candidates
  edm::Handle<reco::JetView> jetView;
  evt.getByToken(jet_token, jetView);
  
  // Convert to a vector of PFJetRefs
  // reco::PFJetRefVector jets = reco::tau::castView<reco::PFJetRefVector>(jetView);
  // edm::RefVector<std::vector<reco::Jet>> jets = reco::tau::castView<edm::RefVector<std::vector<reco::Jet>>>(jetView);
  // edm::RefToBaseVector<reco::Jet> jets = reco::tau::castViewToOtherBase<edm::RefToBaseVector<reco::Jet>>(jetView);
  
  // Get the jet region producer
  edm::Handle<edm::Association<typename TauType::TauJetCollection> > jetRegionHandle;
  evt.getByToken(jetRegion_token, jetRegionHandle);
  
  // Get the charged hadron input collection
  edm::Handle<reco::PFJetChargedHadronAssociation> chargedHadronAssoc;
  evt.getByToken(chargedHadron_token, chargedHadronAssoc);

  // Get the pizero input collection
  edm::Handle<reco::JetPiZeroAssociation> piZeroAssoc;
  evt.getByToken(piZero_token, piZeroAssoc);

  // Update all our builders and modifiers with the event info
  for (typename BuilderList::iterator builder = builders_.begin();
      builder != builders_.end(); ++builder) {
    builder->setup(evt, es);
  }
  for (typename ModifierList::iterator modifier = modifiers_.begin();
      modifier != modifiers_.end(); ++modifier) {
    modifier->setup(evt, es);
  }

  // Create output collection
  auto output = std::make_unique<std::vector<TauType> >();
  output->reserve(jetView->size());
  
  // Loop over the jets and build the taus for each jet
  // BOOST_FOREACH( edm::Ref<std::vector<reco::Jet>> jetRef, jets ) {
  for (size_t i_j = 0; i_j < jetView->size(); ++i_j) {
    const auto& jetRef = jetView->refAt(i_j);

    // Get the jet with extra constituents from an area around the jet
    if(jetRef->pt() - minJetPt_ < 1e-5) continue;
    if(std::abs(jetRef->eta()) - maxJetAbsEta_ > -1e-5) continue;
    auto jetRegionRef = (*jetRegionHandle)[jetRef];
    if ( jetRegionRef.isNull() ) {
      throw cms::Exception("BadJetRegionRef") 
	<< "No jet region can be found for the current jet: " << jetRef.id();
    }
    // Remove all the jet constituents from the jet extras
    std::vector<reco::CandidatePtr> jetCands = jetRef->daughterPtrVector();
    std::vector<reco::CandidatePtr> allRegionalCands = jetRegionRef->daughterPtrVector();
    // Sort both by ref key
    std::sort(jetCands.begin(), jetCands.end());
    std::sort(allRegionalCands.begin(), allRegionalCands.end());
    // Get the regional junk candidates not in the jet.
    std::vector<reco::CandidatePtr> uniqueRegionalCands;

    // This can actually be less than zero, if the jet has really crazy soft
    // stuff really far away from the jet axis.
    if ( allRegionalCands.size() > jetCands.size() ) {
      uniqueRegionalCands.reserve(allRegionalCands.size() - jetCands.size());
    }

    // Subtract the jet cands from the regional cands
    std::set_difference(allRegionalCands.begin(), allRegionalCands.end(),
			jetCands.begin(), jetCands.end(),
			std::back_inserter(uniqueRegionalCands));

    // Get the charged hadrons associated with this jet
    const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons = (*chargedHadronAssoc)[jetRef];

    // Get the pizeros associated with this jet
    const std::vector<reco::RecoTauPiZero>& piZeros = (*piZeroAssoc)[jetRef];
    // Loop over our builders and create the set of taus for this jet
    unsigned int nTausBuilt = 0;
    for (typename BuilderList::const_iterator builder = builders_.begin();
	  builder != builders_.end(); ++builder) {
      // Get a ptr_vector of taus from the builder
      typename reco::tau::RecoTauBuilderPlugin<TauType, PFType>::output_type taus((*builder)(jetRef, chargedHadrons, piZeros, uniqueRegionalCands));
      // JAN - convert reco::Jet ref to PFJet ref (only in direct interaction with PFTau)

      // Make sure all taus have their jetref set correctly
      setJetRefs(taus, jetRef);
      // Copy without selection
      if ( !outputSelector_.get() ) {
        output->insert(output->end(), taus.begin(), taus.end());
        nTausBuilt += taus.size();
      } else {
        // Copy only those that pass the selection.
        BOOST_FOREACH( const TauType& tau, taus ) {
          if ( (*outputSelector_)(tau) ) {
            nTausBuilt++;
            output->push_back(tau);
          }
        }
      }
    }
    // If we didn't build *any* taus for this jet, build a null tau if desired.
    // The null PFTau has no content, but it's four vector is set to that of the
    // jet.
    if ( !nTausBuilt && buildNullTaus_ ) {
      TauType nullTau(buildNullTau(jetRef));
      output->push_back(nullTau);
    }
  }

  // Loop over the taus we have created and apply our modifiers to the taus
  for (typename std::vector<TauType>::iterator tau = output->begin();
	tau != output->end(); ++tau ) {
    for (typename ModifierList::const_iterator modifier = modifiers_.begin();
	  modifier != modifiers_.end(); ++modifier) {
      (*modifier)(*tau);
    }
  }
  
  for (typename ModifierList::iterator modifier = modifiers_.begin();
        modifier != modifiers_.end(); ++modifier) {
    modifier->endEvent();
  }
  
  evt.put(std::move(output));
}

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"

template<>
reco::PFTau RecoGenericTauProducer<reco::PFTau, reco::PFCandidate>::buildNullTau(const edm::RefToBase<reco::Jet>& jetRef) {
  reco::PFTau nullTau(std::numeric_limits<int>::quiet_NaN(), jetRef->p4());
  nullTau.setjetRef(jetRef.castTo<reco::PFJetRef>());
  return std::move(nullTau);
}


template<>
reco::PFBaseTau RecoGenericTauProducer<reco::PFBaseTau, pat::PackedCandidate>::buildNullTau(const edm::RefToBase<reco::Jet>& jetRef) {
  reco::PFBaseTau nullTau(std::numeric_limits<int>::quiet_NaN(), jetRef->p4());
  nullTau.setjetRef(jetRef);
  return std::move(nullTau);
}

template<>
void RecoGenericTauProducer<reco::PFTau, reco::PFCandidate>::setJetRefs(typename reco::tau::RecoTauBuilderPlugin<reco::PFTau, reco::PFCandidate>::output_type taus, const edm::RefToBase<reco::Jet>& jetRef) {
  std::for_each(taus.begin(), taus.end(), boost::bind(&reco::PFTau::setjetRef, _1, jetRef.castTo<reco::PFJetRef>()));
}

template<>
void RecoGenericTauProducer<reco::PFBaseTau, pat::PackedCandidate>::setJetRefs(typename reco::tau::RecoTauBuilderPlugin<reco::PFBaseTau, reco::Candidate>::output_type taus, const edm::RefToBase<reco::Jet>& jetRef) {
  std::for_each(taus.begin(), taus.end(), boost::bind(&reco::PFBaseTau::setjetRef, _1, jetRef));
}

template class RecoGenericTauProducer<reco::PFTau, reco::PFCandidate>; 
template class RecoGenericTauProducer<reco::PFBaseTau, pat::PackedCandidate>; 

typedef RecoGenericTauProducer<reco::PFTau, reco::PFCandidate> RecoTauProducer;
typedef RecoGenericTauProducer<reco::PFBaseTau, pat::PackedCandidate> RecoBaseTauProducer;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauProducer);
DEFINE_FWK_MODULE(RecoBaseTauProducer);
