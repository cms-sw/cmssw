#include <boost/ptr_container/ptr_vector.hpp>

#include <algorithm>
#include <functional>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/TauReco/interface/PFTau.h"

class RecoTauProducer : public edm::EDProducer
{
  public:
    typedef reco::tau::RecoTauBuilderPlugin Builder;
    typedef reco::tau::RecoTauModifierPlugin Modifier;
    typedef boost::ptr_vector<Builder> BuilderList;
    typedef boost::ptr_vector<Modifier> ModifierList;

    explicit RecoTauProducer(const edm::ParameterSet& pset);
    ~RecoTauProducer(){};
    void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    edm::InputTag jetSrc_;
    edm::InputTag piZeroSrc_;
    BuilderList builders_;
    ModifierList modifiers_;
};

RecoTauProducer::RecoTauProducer(const edm::ParameterSet& pset)
{
  jetSrc_ = pset.getParameter<edm::InputTag>("jetSrc");
  piZeroSrc_ = pset.getParameter<edm::InputTag>("piZeroSrc");

  typedef std::vector<edm::ParameterSet> VPSet;
  // Get each of our tau builders
  const VPSet& builders = pset.getParameter<VPSet>("builders");
  for(VPSet::const_iterator builderPSet = builders.begin();
      builderPSet != builders.end(); ++builderPSet) {
    // Get plugin name
    const std::string& pluginType = builderPSet->getParameter<std::string>("plugin");
    // Build the plugin
    builders_.push_back(RecoTauBuilderPluginFactory::get()->create(pluginType, *builderPSet));
  }

  const VPSet& modfiers = pset.getParameter<VPSet>("modifiers");
  for(VPSet::const_iterator modfierPSet = modfiers.begin();
      modfierPSet != modfiers.end(); ++modfierPSet)
  {
    // Get plugin name
    const std::string& pluginType = modfierPSet->getParameter<std::string>("plugin");
    // Build the plugin
    modifiers_.push_back(RecoTauModifierPluginFactory::get()->create(pluginType, *modfierPSet));
  }

  produces<reco::PFTauCollection>();
}

void RecoTauProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  // Get the jet input collection
  edm::Handle<reco::PFJetCollection> pfJets;
  evt.getByLabel(jetSrc_, pfJets);

  // Get the pizero input collection
  edm::Handle<reco::JetPiZeroAssociation> piZeroAssoc;
  evt.getByLabel(piZeroSrc_, piZeroAssoc);

  // Update all our builders and modifiers with the event info
  for(BuilderList::iterator builder = builders_.begin();
      builder != builders_.end(); ++builder) {
    builder->setup(evt, es);
  }
  for(ModifierList::iterator modifier = modifiers_.begin();
      modifier != modifiers_.end(); ++modifier) {
    modifier->setup(evt, es);
  }

  // Create output collection
  std::auto_ptr<reco::PFTauCollection> output(new reco::PFTauCollection());


  // Loop over the jets and build the taus for each jet
  const size_t nJets = pfJets->size();
  for(size_t iJet = 0; iJet < nJets; ++iJet)
  {
    // Get a Ref to the PFJet
    const reco::PFJetRef jetRef(pfJets, iJet);

    // Get the PiZeros associated with this jet
    const std::vector<reco::RecoTauPiZero>& piZeros = (*piZeroAssoc)[jetRef];

    // Loop over our builders and create the set of taus for this jet
    for(BuilderList::const_iterator builder = builders_.begin();
        builder != builders_.end(); ++builder)
    {
      std::vector<reco::PFTau> taus((*builder)(jetRef, piZeros));
      // Ensure the jetRef is set correctly
      output->insert(output->end(), taus.begin(), taus.end());
    }

  }

  // Loop over the taus we have created and apply our modifiers to the taus
  for(reco::PFTauCollection::iterator tau = output->begin();
      tau != output->end(); ++tau) {
    for(ModifierList::const_iterator modifier = modifiers_.begin();
        modifier != modifiers_.end(); ++modifier) {
      (*modifier)(*tau);
    }
  }

  evt.put(output);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauProducer);
