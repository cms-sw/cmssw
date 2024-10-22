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

#include <algorithm>
#include <functional>
#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFJetChargedHadronAssociation.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/Common/interface/Association.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class RecoTauProducer : public edm::stream::EDProducer<> {
public:
  typedef reco::tau::RecoTauBuilderPlugin Builder;
  typedef reco::tau::RecoTauModifierPlugin Modifier;
  typedef std::vector<std::unique_ptr<Builder>> BuilderList;
  typedef std::vector<std::unique_ptr<Modifier>> ModifierList;

  explicit RecoTauProducer(const edm::ParameterSet& pset);
  ~RecoTauProducer() override {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag jetSrc_;
  edm::InputTag jetRegionSrc_;
  edm::InputTag chargedHadronSrc_;
  edm::InputTag piZeroSrc_;

  double minJetPt_;
  double maxJetAbsEta_;
  //token definition
  edm::EDGetTokenT<reco::JetView> jet_token;
  edm::EDGetTokenT<edm::AssociationMap<edm::OneToOne<reco::JetView, reco::JetView>>> jetRegion_token;
  edm::EDGetTokenT<reco::PFJetChargedHadronAssociation> chargedHadron_token;
  edm::EDGetTokenT<reco::JetPiZeroAssociation> piZero_token;

  BuilderList builders_;
  ModifierList modifiers_;
  // Optional selection on the output of the taus
  std::unique_ptr<StringCutObjectSelector<reco::PFTau>> outputSelector_;
  // Whether or not to add build a tau from a jet for which the builders
  // return no taus.  The tau will have no content, only the four vector of
  // the orginal jet.
  bool buildNullTaus_;
};

RecoTauProducer::RecoTauProducer(const edm::ParameterSet& pset) {
  jetSrc_ = pset.getParameter<edm::InputTag>("jetSrc");
  jetRegionSrc_ = pset.getParameter<edm::InputTag>("jetRegionSrc");
  chargedHadronSrc_ = pset.getParameter<edm::InputTag>("chargedHadronSrc");
  piZeroSrc_ = pset.getParameter<edm::InputTag>("piZeroSrc");

  minJetPt_ = pset.getParameter<double>("minJetPt");
  maxJetAbsEta_ = pset.getParameter<double>("maxJetAbsEta");
  //consumes definition
  jet_token = consumes<reco::JetView>(jetSrc_);
  jetRegion_token = consumes<edm::AssociationMap<edm::OneToOne<reco::JetView, reco::JetView>>>(jetRegionSrc_);
  chargedHadron_token = consumes<reco::PFJetChargedHadronAssociation>(chargedHadronSrc_);
  piZero_token = consumes<reco::JetPiZeroAssociation>(piZeroSrc_);

  typedef std::vector<edm::ParameterSet> VPSet;
  // Get each of our tau builders
  const VPSet& builders = pset.getParameter<VPSet>("builders");
  for (VPSet::const_iterator builderPSet = builders.begin(); builderPSet != builders.end(); ++builderPSet) {
    // Get plugin name
    const std::string& pluginType = builderPSet->getParameter<std::string>("plugin");
    // Build the plugin
    builders_.emplace_back(RecoTauBuilderPluginFactory::get()->create(pluginType, *builderPSet, consumesCollector()));
  }

  const VPSet& modfiers = pset.getParameter<VPSet>("modifiers");
  for (VPSet::const_iterator modfierPSet = modfiers.begin(); modfierPSet != modfiers.end(); ++modfierPSet) {
    // Get plugin name
    const std::string& pluginType = modfierPSet->getParameter<std::string>("plugin");
    // Build the plugin
    modifiers_.emplace_back(RecoTauModifierPluginFactory::get()->create(pluginType, *modfierPSet, consumesCollector()));
  }

  // Check if we want to apply a final output selection
  std::string selection = pset.getParameter<std::string>("outputSelection");
  if (!selection.empty()) {
    outputSelector_ = std::make_unique<StringCutObjectSelector<reco::PFTau>>(selection);
  }
  buildNullTaus_ = pset.getParameter<bool>("buildNullTaus");

  produces<reco::PFTauCollection>();
}

void RecoTauProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get the jet input collection via a view of Candidates
  edm::Handle<reco::JetView> jetView;
  evt.getByToken(jet_token, jetView);

  // Get the jet region producer
  edm::Handle<edm::AssociationMap<edm::OneToOne<reco::JetView, reco::JetView>>> jetRegionHandle;
  evt.getByToken(jetRegion_token, jetRegionHandle);

  // Get the charged hadron input collection
  edm::Handle<reco::PFJetChargedHadronAssociation> chargedHadronAssoc;
  evt.getByToken(chargedHadron_token, chargedHadronAssoc);

  // Get the pizero input collection
  edm::Handle<reco::JetPiZeroAssociation> piZeroAssoc;
  evt.getByToken(piZero_token, piZeroAssoc);

  // Update all our builders and modifiers with the event info
  for (auto& builder : builders_) {
    builder->setup(evt, es);
  }
  for (auto& modifier : modifiers_) {
    modifier->setup(evt, es);
  }

  // Create output collection
  auto output = std::make_unique<reco::PFTauCollection>();
  output->reserve(jetView->size());

  // Loop over the jets and build the taus for each jet
  for (size_t i_j = 0; i_j < jetView->size(); ++i_j) {
    const auto& jetRef = jetView->refAt(i_j);
    // Get the jet with extra constituents from an area around the jet
    if (jetRef->pt() - minJetPt_ < 1e-5)
      continue;
    if (std::abs(jetRef->eta()) - maxJetAbsEta_ > -1e-5)
      continue;
    reco::JetBaseRef jetRegionRef = (*jetRegionHandle)[jetRef];
    if (jetRegionRef.isNull()) {
      throw cms::Exception("BadJetRegionRef") << "No jet region can be found for the current jet: " << jetRef.id();
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
    if (allRegionalCands.size() > jetCands.size()) {
      uniqueRegionalCands.reserve(allRegionalCands.size() - jetCands.size());
    }

    // Subtract the jet cands from the regional cands
    std::set_difference(allRegionalCands.begin(),
                        allRegionalCands.end(),
                        jetCands.begin(),
                        jetCands.end(),
                        std::back_inserter(uniqueRegionalCands));

    // Get the charged hadrons associated with this jet
    const std::vector<reco::PFRecoTauChargedHadron>& chargedHadrons = (*chargedHadronAssoc)[jetRef];

    // Get the pizeros associated with this jet
    const std::vector<reco::RecoTauPiZero>& piZeros = (*piZeroAssoc)[jetRef];
    // Loop over our builders and create the set of taus for this jet
    unsigned int nTausBuilt = 0;
    for (const auto& builder : builders_) {
      // Get a std::vector of std::unique_ptr to taus from the builder
      reco::tau::RecoTauBuilderPlugin::output_type taus(
          (*builder)(jetRef, chargedHadrons, piZeros, uniqueRegionalCands));

      // Make sure all taus have their jetref set correctly
      std::for_each(taus.begin(), taus.end(), [&](auto& arg) { arg->setjetRef(reco::JetBaseRef(jetRef)); });
      // Copy without selection
      if (!outputSelector_.get()) {
        for (auto& tau : taus) {
          output->push_back(*tau);
        }
        nTausBuilt += taus.size();
      } else {
        // Copy only those that pass the selection.
        for (auto const& tau : taus) {
          if ((*outputSelector_)(*tau)) {
            nTausBuilt++;
            output->push_back(*tau);
          }
        }
      }
    }
    // If we didn't build *any* taus for this jet, build a null tau if desired.
    // The null PFTau has no content, but it's four vector is set to that of the
    // jet.
    if (!nTausBuilt && buildNullTaus_) {
      reco::PFTau nullTau(std::numeric_limits<int>::quiet_NaN(), jetRef->p4());
      nullTau.setjetRef(reco::JetBaseRef(jetRef));
      output->push_back(nullTau);
    }
  }

  // Loop over the taus we have created and apply our modifiers to the taus
  for (auto& tau : *output) {
    for (const auto& modifier : modifiers_) {
      (*modifier)(tau);
    }
  }

  for (auto& modifier : modifiers_) {
    modifier->endEvent();
  }

  evt.put(std::move(output));
}

void RecoTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // combinatoricRecoTaus
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("piZeroSrc", edm::InputTag("ak4PFJetsRecoTauPiZeros"));

  edm::ParameterSetDescription desc_qualityCuts;
  reco::tau::RecoTauQualityCuts::fillDescriptions(desc_qualityCuts);

  {
    edm::ParameterSetDescription vpsd_modifiers;
    vpsd_modifiers.add<std::string>("name");
    vpsd_modifiers.add<std::string>("plugin");
    vpsd_modifiers.add<int>("verbosity", 0);

    vpsd_modifiers.add<edm::ParameterSetDescription>("qualityCuts", desc_qualityCuts);
    vpsd_modifiers.addOptional<edm::InputTag>("ElectronPreIDProducer");
    vpsd_modifiers.addOptional<std::string>("DataType");
    vpsd_modifiers.addOptional<double>("maximumForElectrionPreIDOutput");
    vpsd_modifiers.addOptional<double>("ElecPreIDLeadTkMatch_maxDR");
    vpsd_modifiers.addOptional<double>("EcalStripSumE_minClusEnergy");
    vpsd_modifiers.addOptional<double>("EcalStripSumE_deltaPhiOverQ_minValue");
    vpsd_modifiers.addOptional<double>("EcalStripSumE_deltaPhiOverQ_maxValue");
    vpsd_modifiers.addOptional<double>("EcalStripSumE_deltaEta");
    vpsd_modifiers.addOptional<double>("dRaddNeutralHadron");
    vpsd_modifiers.addOptional<double>("minGammaEt");
    vpsd_modifiers.addOptional<double>("dRaddPhoton");
    vpsd_modifiers.addOptional<double>("minNeutralHadronEt");
    vpsd_modifiers.addOptional<edm::InputTag>("pfTauTagInfoSrc");
    vpsd_modifiers.addOptional<edm::InputTag>("trackSrc");

    desc.addVPSet("modifiers", vpsd_modifiers);
  }

  desc.add<edm::InputTag>("jetRegionSrc", edm::InputTag("recoTauAK4PFJets08Region"));
  desc.add<double>("maxJetAbsEta", 2.5);
  desc.add<std::string>("outputSelection", "leadPFChargedHadrCand().isNonnull()");
  desc.add<edm::InputTag>("chargedHadronSrc", edm::InputTag("ak4PFJetsRecoTauChargedHadrons"));
  desc.add<double>("minJetPt", 14.0);
  desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJets"));

  {
    edm::ParameterSetDescription desc_builders;
    desc_builders.add<std::string>("name");
    desc_builders.add<std::string>("plugin");
    desc_builders.add<int>("verbosity", 0);

    desc_builders.add<edm::ParameterSetDescription>("qualityCuts", desc_qualityCuts);
    {
      edm::ParameterSetDescription desc_decayModes;
      desc_decayModes.add<unsigned int>("nPiZeros", 0);
      desc_decayModes.add<unsigned int>("maxPiZeros", 0);
      desc_decayModes.add<unsigned int>("nCharged", 1);
      desc_decayModes.add<unsigned int>("maxTracks", 6);
      desc_builders.addVPSetOptional("decayModes", desc_decayModes);
    }
    desc_builders.add<double>("minAbsPhotonSumPt_insideSignalCone", 2.5);
    desc_builders.add<double>("minRelPhotonSumPt_insideSignalCone", 0.1);
    desc_builders.add<edm::InputTag>("pfCandSrc", edm::InputTag("particleFlow"));

    desc_builders.addOptional<std::string>("signalConeSize");
    desc_builders.addOptional<double>("isolationConeSize");
    desc_builders.addOptional<double>("minAbsPhotonSumPt_outsideSignalCone");
    desc_builders.addOptional<double>("minRelPhotonSumPt_outsideSignalCone");
    desc_builders.addOptional<std::string>("isoConeChargedHadrons");
    desc_builders.addOptional<std::string>("isoConeNeutralHadrons");
    desc_builders.addOptional<std::string>("isoConePiZeros");
    desc_builders.addOptional<double>("leadObjectPt");
    desc_builders.addOptional<std::string>("matchingCone");
    desc_builders.addOptional<int>("maxSignalConeChargedHadrons");
    desc_builders.addOptional<std::string>("signalConeChargedHadrons");
    desc_builders.addOptional<std::string>("signalConeNeutralHadrons");
    desc_builders.addOptional<std::string>("signalConePiZeros");
    desc_builders.addOptional<bool>("usePFLeptons");

    std::vector<edm::ParameterSet> vpset_default;
    {
      edm::ParameterSet pset_default_builders;
      pset_default_builders.addParameter<std::string>("name", "");
      pset_default_builders.addParameter<std::string>("plugin", "");
      pset_default_builders.addParameter<int>("verbosity", 0);
      pset_default_builders.addParameter<double>("minAbsPhotonSumPt_insideSignalCone", 2.5);
      pset_default_builders.addParameter<double>("minRelPhotonSumPt_insideSignalCone", 0.1);
      pset_default_builders.addParameter<edm::InputTag>("pfCandSrc", edm::InputTag("particleFlow"));
      vpset_default.push_back(pset_default_builders);
    }
    desc.addVPSet("builders", desc_builders, vpset_default);
  }

  desc.add<bool>("buildNullTaus", false);
  desc.add<int>("verbosity", 0);
  descriptions.add("combinatoricRecoTaus", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauProducer);
