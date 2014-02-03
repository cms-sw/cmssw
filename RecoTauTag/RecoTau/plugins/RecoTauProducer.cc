/*
 * RecoTauProducer
 *
 * Interface between the various tau algorithms and the edm::Event.  The
 * RecoTauProducer takes as data input is a collection (view) of reco::PFJets,
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
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/foreach.hpp>

#include <algorithm>
#include <functional>

#include "FWCore/Framework/interface/EDProducer.h"
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
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Association.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class RecoTauProducer : public edm::EDProducer 
{
 public:
  typedef reco::tau::RecoTauBuilderPlugin Builder;
  typedef reco::tau::RecoTauModifierPlugin Modifier;
  typedef boost::ptr_vector<Builder> BuilderList;
  typedef boost::ptr_vector<Modifier> ModifierList;

  explicit RecoTauProducer(const edm::ParameterSet& pset);
  ~RecoTauProducer() {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

 private:
  edm::InputTag jetSrc_;
  edm::InputTag jetRegionSrc_;
  edm::InputTag chargedHadronSrc_;
  edm::InputTag piZeroSrc_;
 //token definition
  edm::EDGetTokenT<reco::CandidateView> jet_token;
  edm::EDGetTokenT<edm::Association<reco::PFJetCollection> > jetRegion_token;
  edm::EDGetTokenT<reco::PFJetChargedHadronAssociation> chargedHadron_token;
  edm::EDGetTokenT<reco::JetPiZeroAssociation> piZero_token;

  BuilderList builders_;
  ModifierList modifiers_;
  // Optional selection on the output of the taus
  std::auto_ptr<StringCutObjectSelector<reco::PFTau> > outputSelector_;
  // Whether or not to add build a tau from a jet for which the builders
  // return no taus.  The tau will have no content, only the four vector of
  // the orginal jet.
  bool buildNullTaus_;
};

RecoTauProducer::RecoTauProducer(const edm::ParameterSet& pset) 
{
  jetSrc_ = pset.getParameter<edm::InputTag>("jetSrc");
  jetRegionSrc_ = pset.getParameter<edm::InputTag>("jetRegionSrc");
  chargedHadronSrc_ = pset.getParameter<edm::InputTag>("chargedHadronSrc");
  piZeroSrc_ = pset.getParameter<edm::InputTag>("piZeroSrc");
  
  //consumes definition
  jet_token=consumes<reco::CandidateView>(jetSrc_);
  jetRegion_token = consumes<edm::Association<reco::PFJetCollection> >(jetRegionSrc_);
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
    builders_.push_back(RecoTauBuilderPluginFactory::get()->create(pluginType, *builderPSet, consumesCollector()));
  }

  const VPSet& modfiers = pset.getParameter<VPSet>("modifiers");
  for ( VPSet::const_iterator modfierPSet = modfiers.begin();
	modfierPSet != modfiers.end(); ++modfierPSet) {
    // Get plugin name
    const std::string& pluginType = modfierPSet->getParameter<std::string>("plugin");
    // Build the plugin
    reco::tau::RecoTauModifierPlugin* plugin = 0;
    plugin = RecoTauModifierPluginFactory::get()->create(pluginType, *modfierPSet, consumesCollector());
    plugin->beginJob(this);
    modifiers_.push_back(plugin);
  }

  // Check if we want to apply a final output selection
  if ( pset.exists("outputSelection") ) {
    std::string selection = pset.getParameter<std::string>("outputSelection");
    if ( selection != "" ) {
      outputSelector_.reset(new StringCutObjectSelector<reco::PFTau>(selection));
    }
  }
  buildNullTaus_ = pset.getParameter<bool>("buildNullTaus");

  produces<reco::PFTauCollection>();
}

void RecoTauProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  // Get the jet input collection via a view of Candidates
  edm::Handle<reco::CandidateView> jetView;
  evt.getByToken(jet_token, jetView);
  
  // Convert to a vector of PFJetRefs
  reco::PFJetRefVector jets = reco::tau::castView<reco::PFJetRefVector>(jetView);
  
  // Get the jet region producer
  edm::Handle<edm::Association<reco::PFJetCollection> > jetRegionHandle;
  evt.getByToken(jetRegion_token, jetRegionHandle);
  
  // Get the charged hadron input collection
  edm::Handle<reco::PFJetChargedHadronAssociation> chargedHadronAssoc;
  evt.getByToken(chargedHadron_token, chargedHadronAssoc);

  // Get the pizero input collection
  edm::Handle<reco::JetPiZeroAssociation> piZeroAssoc;
  evt.getByToken(piZero_token, piZeroAssoc);

  // Update all our builders and modifiers with the event info
  for (BuilderList::iterator builder = builders_.begin();
      builder != builders_.end(); ++builder) {
    builder->setup(evt, es);
  }
  for (ModifierList::iterator modifier = modifiers_.begin();
      modifier != modifiers_.end(); ++modifier) {
    modifier->setup(evt, es);
  }

  // Create output collection
  std::auto_ptr<reco::PFTauCollection> output(new reco::PFTauCollection());
  output->reserve(jets.size());
  
  // Loop over the jets and build the taus for each jet
  BOOST_FOREACH( reco::PFJetRef jetRef, jets ) {
    // Get the jet with extra constituents from an area around the jet
    reco::PFJetRef jetRegionRef = (*jetRegionHandle)[jetRef];
    if ( jetRegionRef.isNull() ) {
      throw cms::Exception("BadJetRegionRef") 
	<< "No jet region can be found for the current jet: " << jetRef.id();
    }
    // Remove all the jet constituents from the jet extras
    std::vector<reco::PFCandidatePtr> jetCands = jetRef->getPFConstituents();
    std::vector<reco::PFCandidatePtr> allRegionalCands = jetRegionRef->getPFConstituents();
    // Sort both by ref key
    std::sort(jetCands.begin(), jetCands.end());
    std::sort(allRegionalCands.begin(), allRegionalCands.end());
    // Get the regional junk candidates not in the jet.
    std::vector<reco::PFCandidatePtr> uniqueRegionalCands;

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
    for ( BuilderList::const_iterator builder = builders_.begin();
	  builder != builders_.end(); ++builder) {
      // Get a ptr_vector of taus from the builder
      reco::tau::RecoTauBuilderPlugin::output_type taus((*builder)(jetRef, chargedHadrons, piZeros, uniqueRegionalCands));
      // Make sure all taus have their jetref set correctly
      std::for_each(taus.begin(), taus.end(), boost::bind(&reco::PFTau::setjetRef, _1, jetRef));
      // Copy without selection
      if ( !outputSelector_.get() ) {
        output->insert(output->end(), taus.begin(), taus.end());
        nTausBuilt += taus.size();
      } else {
        // Copy only those that pass the selection.
        BOOST_FOREACH( const reco::PFTau& tau, taus ) {
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
      reco::PFTau nullTau(std::numeric_limits<int>::quiet_NaN(), jetRef->p4());
      nullTau.setjetRef(jetRef);
      output->push_back(nullTau);
    }
  }

  // Loop over the taus we have created and apply our modifiers to the taus
  for ( reco::PFTauCollection::iterator tau = output->begin();
	tau != output->end(); ++tau ) {
    for ( ModifierList::const_iterator modifier = modifiers_.begin();
	  modifier != modifiers_.end(); ++modifier ) {
      (*modifier)(*tau);
    }
  }
  
  for ( ModifierList::iterator modifier = modifiers_.begin();
        modifier != modifiers_.end(); ++modifier ) {
    modifier->endEvent();
  }
  
  evt.put(output);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauProducer);
