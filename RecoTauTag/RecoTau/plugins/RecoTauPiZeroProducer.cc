/*
 * RecoTauPiZeroProducer
 *
 * Author: Evan K. Friis, UC Davis
 *
 * Associates reconstructed PiZeros to PFJets.  The PiZeros are built using one
 * or more RecoTauBuilder plugins.  Any overlaps (PiZeros sharing constituents)
 * are removed, with the best PiZero candidates taken.  The 'best' are defined
 * via the input list of RecoTauPiZeroQualityPlugins, which form a
 * lexicograpical ranking.
 *
 */

#include <algorithm>
#include <boost/ptr_container/ptr_list.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <functional>
#include <memory>

        
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCleaningTools.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/Common/interface/Association.h"

#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class RecoTauPiZeroProducer : public edm::stream::EDProducer<> {
public:
  typedef reco::tau::RecoTauPiZeroBuilderPlugin Builder;
  typedef reco::tau::RecoTauPiZeroQualityPlugin Ranker;

  explicit RecoTauPiZeroProducer(const edm::ParameterSet& pset);
  ~RecoTauPiZeroProducer() override {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  void print(const std::vector<reco::RecoTauPiZero>& piZeros, std::ostream& out);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef boost::ptr_vector<Builder> builderList;
  typedef boost::ptr_vector<Ranker> rankerList;
  typedef boost::ptr_vector<reco::RecoTauPiZero> PiZeroVector;
  typedef boost::ptr_list<reco::RecoTauPiZero> PiZeroList;

  typedef reco::tau::RecoTauLexicographicalRanking<rankerList, reco::RecoTauPiZero> PiZeroPredicate;

  builderList builders_;
  rankerList rankers_;
  std::unique_ptr<PiZeroPredicate> predicate_;
  double piZeroMass_;

  // Output selector
  std::unique_ptr<StringCutObjectSelector<reco::RecoTauPiZero>> outputSelector_;

  //consumes interface
  edm::EDGetTokenT<reco::JetView> cand_token;

  double minJetPt_;
  double maxJetAbsEta_;

  int verbosity_;
};

RecoTauPiZeroProducer::RecoTauPiZeroProducer(const edm::ParameterSet& pset) {
  cand_token = consumes<reco::JetView>(pset.getParameter<edm::InputTag>("jetSrc"));
  minJetPt_ = pset.getParameter<double>("minJetPt");
  maxJetAbsEta_ = pset.getParameter<double>("maxJetAbsEta");

  typedef std::vector<edm::ParameterSet> VPSet;
  // Get the mass hypothesis for the pizeros
  piZeroMass_ = pset.getParameter<double>("massHypothesis");

  // Get each of our PiZero builders
  const VPSet& builders = pset.getParameter<VPSet>("builders");

  for (VPSet::const_iterator builderPSet = builders.begin(); builderPSet != builders.end(); ++builderPSet) {
    // Get plugin name
    const std::string& pluginType = builderPSet->getParameter<std::string>("plugin");
    // Build the plugin
    builders_.push_back(
        RecoTauPiZeroBuilderPluginFactory::get()->create(pluginType, *builderPSet, consumesCollector()));
  }

  // Get each of our quality rankers
  const VPSet& rankers = pset.getParameter<VPSet>("ranking");
  for (VPSet::const_iterator rankerPSet = rankers.begin(); rankerPSet != rankers.end(); ++rankerPSet) {
    const std::string& pluginType = rankerPSet->getParameter<std::string>("plugin");
    rankers_.push_back(RecoTauPiZeroQualityPluginFactory::get()->create(pluginType, *rankerPSet));
  }

  // Build the sorting predicate
  predicate_ = std::make_unique<PiZeroPredicate>(rankers_);

  // now all producers apply a final output selection
  std::string selection = pset.getParameter<std::string>("outputSelection");
  if (!selection.empty()) {
    outputSelector_ = std::make_unique<StringCutObjectSelector<reco::RecoTauPiZero>>(selection);
  }

  verbosity_ = pset.getParameter<int>("verbosity");

  produces<reco::JetPiZeroAssociation>();
}

void RecoTauPiZeroProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get a view of our jets via the base candidates
  edm::Handle<reco::JetView> jetView;
  evt.getByToken(cand_token, jetView);

  // Give each of our plugins a chance at doing something with the edm::Event
  for (auto& builder : builders_) {
    builder.setup(evt, es);
  }

  // Make our association
  std::unique_ptr<reco::JetPiZeroAssociation> association;

  association = std::make_unique<reco::JetPiZeroAssociation>(reco::JetRefBaseProd(jetView));

  // Loop over our jets
  size_t nJets = jetView->size();
  for (size_t i = 0; i < nJets; ++i) {
    const reco::JetBaseRef jet(jetView->refAt(i));

    if (jet->pt() - minJetPt_ < 1e-5)
      continue;
    if (std::abs(jet->eta()) - maxJetAbsEta_ > -1e-5)
      continue;
    // Build our global list of RecoTauPiZero
    PiZeroList dirtyPiZeros;

    // Compute the pi zeros from this jet for all the desired algorithms
    for (auto const& builder : builders_) {
      try {
        PiZeroVector result(builder(*jet));
        dirtyPiZeros.transfer(dirtyPiZeros.end(), result);
      } catch (cms::Exception& exception) {
        edm::LogError("BuilderPluginException")
            << "Exception caught in builder plugin " << builder.name() << ", rethrowing" << std::endl;
        throw exception;
      }
    }
    // Rank the candidates according to our quality plugins
    dirtyPiZeros.sort(*predicate_);

    // Keep track of the photons in the clean collection
    std::vector<reco::RecoTauPiZero> cleanPiZeros;
    std::set<reco::CandidatePtr> photonsInCleanCollection;
    while (!dirtyPiZeros.empty()) {
      // Pull our candidate pi zero from the front of the list
      std::unique_ptr<reco::RecoTauPiZero> toAdd(dirtyPiZeros.pop_front().release());
      // If this doesn't pass our basic selection, discard it.
      if (!(*outputSelector_)(*toAdd)) {
        continue;
      }
      // Find the sub-gammas that are not already in the cleaned collection
      std::vector<reco::CandidatePtr> uniqueGammas;
      std::set_difference(toAdd->daughterPtrVector().begin(),
                          toAdd->daughterPtrVector().end(),
                          photonsInCleanCollection.begin(),
                          photonsInCleanCollection.end(),
                          std::back_inserter(uniqueGammas));
      // If the pi zero has no unique gammas, discard it.  Note toAdd is deleted
      // when it goes out of scope.
      if (uniqueGammas.empty()) {
        continue;
      } else if (uniqueGammas.size() == toAdd->daughterPtrVector().size()) {
        // Check if it is composed entirely of unique gammas.  In this case
        // immediately add it to the clean collection.
        photonsInCleanCollection.insert(toAdd->daughterPtrVector().begin(), toAdd->daughterPtrVector().end());
        cleanPiZeros.push_back(*toAdd);
      } else {
        // Otherwise update the pizero that contains only the unique gammas and
        // add it back into the sorted list of dirty PiZeros
        toAdd->clearDaughters();
        // Add each of the unique daughters back to the pizero
        for (auto const& gamma : uniqueGammas) {
          toAdd->addDaughter(gamma);
        }
        // Update the four vector
        AddFourMomenta p4Builder_;
        p4Builder_.set(*toAdd);
        // Put this pi zero back into the collection of sorted dirty pizeros
        PiZeroList::iterator insertionPoint =
            std::lower_bound(dirtyPiZeros.begin(), dirtyPiZeros.end(), *toAdd, *predicate_);
        dirtyPiZeros.insert(insertionPoint, std::move(toAdd));
      }
    }
    // Apply the mass hypothesis if desired
    if (piZeroMass_ >= 0) {
      for (auto& cleanPiZero : cleanPiZeros) {
        cleanPiZero.setMass(this->piZeroMass_);
      };
    }
    // Add to association
    if (verbosity_ >= 2) {
      print(cleanPiZeros, std::cout);
    }
    association->setValue(jet.key(), cleanPiZeros);
  }
  evt.put(std::move(association));
}

// Print some helpful information
void RecoTauPiZeroProducer::print(const std::vector<reco::RecoTauPiZero>& piZeros, std::ostream& out) {
  const unsigned int width = 25;
  for (auto const& piZero : piZeros) {
    out << piZero;
    out << "* Rankers:" << std::endl;
    for (rankerList::const_iterator ranker = rankers_.begin(); ranker != rankers_.end(); ++ranker) {
      out << "* " << std::setiosflags(std::ios::left) << std::setw(width) << ranker->name() << " "
          << std::resetiosflags(std::ios::left) << std::setprecision(3) << (*ranker)(piZero);
      out << std::endl;
    }
  }
}

void RecoTauPiZeroProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // common parameter descriptions
  edm::ParameterSetDescription desc_ranking;
  desc_ranking.add<std::string>("selectionPassFunction", "Func");
  desc_ranking.add<double>("selectionFailValue", 1000);
  desc_ranking.add<std::string>("selection", "Sel");
  desc_ranking.add<std::string>("name", "name");
  desc_ranking.add<std::string>("plugin", "plugin");
  edm::ParameterSet pset_ranking;
  pset_ranking.addParameter<std::string>("selectionPassFunction", "");
  pset_ranking.addParameter<double>("selectionFailValue", 1000);
  pset_ranking.addParameter<std::string>("selection", "");
  pset_ranking.addParameter<std::string>("name", "");
  pset_ranking.addParameter<std::string>("plugin", "");
  std::vector<edm::ParameterSet> vpsd_ranking;
  vpsd_ranking.push_back(pset_ranking);

  edm::ParameterSetDescription desc_signalQualityCuts;
  desc_signalQualityCuts.add<double>("maxDeltaZ", 0.4);
  desc_signalQualityCuts.add<double>("minTrackPt", 0.5);
  desc_signalQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  desc_signalQualityCuts.add<double>("maxTrackChi2", 100.0);
  desc_signalQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  desc_signalQualityCuts.add<double>("minGammaEt", 1.0);
  desc_signalQualityCuts.add<unsigned int>("minTrackHits", 3);
  desc_signalQualityCuts.addOptional<double>("minNeutralHadronEt");
  desc_signalQualityCuts.add<double>("maxTransverseImpactParameter", 0.1);
  desc_signalQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription desc_vxAssocQualityCuts;
  desc_vxAssocQualityCuts.add<double>("minTrackPt", 0.5);
  desc_vxAssocQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  desc_vxAssocQualityCuts.add<double>("maxTrackChi2", 100.0);
  desc_vxAssocQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  desc_vxAssocQualityCuts.add<double>("minGammaEt", 1.0);
  desc_vxAssocQualityCuts.add<unsigned int>("minTrackHits", 3);
  desc_vxAssocQualityCuts.add<double>("maxTransverseImpactParameter", 0.1);
  desc_vxAssocQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription desc_isolationQualityCuts;
  desc_isolationQualityCuts.add<double>("maxDeltaZ", 0.2);
  desc_isolationQualityCuts.add<double>("minTrackPt", 1.0);
  desc_isolationQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  desc_isolationQualityCuts.add<double>("maxTrackChi2", 100.0);
  desc_isolationQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  desc_isolationQualityCuts.add<double>("minGammaEt", 1.5);
  desc_isolationQualityCuts.add<unsigned int>("minTrackHits", 8);
  desc_isolationQualityCuts.add<double>("maxTransverseImpactParameter", 0.03);
  desc_isolationQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription desc_qualityCuts;
  desc_qualityCuts.add<edm::ParameterSetDescription>("signalQualityCuts", desc_signalQualityCuts);
  desc_qualityCuts.add<edm::ParameterSetDescription>("vxAssocQualityCuts", desc_vxAssocQualityCuts);
  desc_qualityCuts.add<edm::ParameterSetDescription>("isolationQualityCuts", desc_isolationQualityCuts);
  desc_qualityCuts.add<std::string>("leadingTrkOrPFCandOption", "leadPFCand");
  desc_qualityCuts.add<std::string>("pvFindingAlgo", "closestInDeltaZ");
  desc_qualityCuts.add<edm::InputTag>("primaryVertexSrc", edm::InputTag("offlinePrimaryVertices"));
  desc_qualityCuts.add<bool>("vertexTrackFiltering", false);
  desc_qualityCuts.add<bool>("recoverLeadingTrk", false);

  edm::ParameterSet pset_builders;
  pset_builders.addParameter<std::string>("name", "");
  pset_builders.addParameter<std::string>("plugin", "");
  edm::ParameterSet qualityCuts;
  pset_builders.addParameter<edm::ParameterSet>("qualityCuts", qualityCuts);
  pset_builders.addParameter<int>("verbosity", 0);

  {
    // Tailored on ak4PFJetsLegacyHPSPiZeros
    edm::ParameterSetDescription desc;
    desc.add<double>("massHypothesis", 0.136);
    desc.addVPSet("ranking", desc_ranking, vpsd_ranking);
    desc.add<int>("verbosity", 0);
    desc.add<double>("maxJetAbsEta", 2.5);
    desc.add<std::string>("outputSelection", "pt > 0");
    desc.add<double>("minJetPt", 14.0);
    desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJets"));

    edm::ParameterSetDescription desc_builders;
    {
      edm::ParameterSetDescription psd0;
      psd0.add<std::string>("function", "TMath::Min(0.3, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))");
      psd0.add<double>("par1", 0.707716);
      psd0.add<double>("par0", 0.352476);
      desc_builders.addOptional<edm::ParameterSetDescription>("stripPhiAssociationDistanceFunc", psd0);
    }
    {
      edm::ParameterSetDescription psd0;
      psd0.add<std::string>("function", "TMath::Min(0.15, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))");
      psd0.add<double>("par1", 0.658701);
      psd0.add<double>("par0", 0.197077);
      desc_builders.addOptional<edm::ParameterSetDescription>("stripEtaAssociationDistanceFunc", psd0);
    }
    desc_builders.addOptional<double>("stripEtaAssociationDistance", 0.05);
    desc_builders.addOptional<double>("stripPhiAssociationDistance", 0.2);

    desc_builders.add<edm::ParameterSetDescription>("qualityCuts", desc_qualityCuts);

    desc_builders.add<std::string>("name");
    desc_builders.add<std::string>("plugin");
    desc_builders.add<int>("verbosity", 0);

    desc_builders.addOptional<bool>("makeCombinatoricStrips");
    desc_builders.addOptional<int>("maxStripBuildIterations");
    desc_builders.addOptional<double>("minGammaEtStripAdd");
    desc_builders.addOptional<double>("minGammaEtStripSeed");
    desc_builders.addOptional<double>("minStripEt");
    desc_builders.addOptional<std::vector<int>>("stripCandidatesParticleIds");
    desc_builders.addOptional<bool>("updateStripAfterEachDaughter");
    desc_builders.addOptional<bool>("applyElecTrackQcuts");

    std::vector<edm::ParameterSet> vpsd_builders;
    vpsd_builders.push_back(pset_builders);
    desc.addVPSet("builders", desc_builders, vpsd_builders);

    descriptions.add("recoTauPiZeroProducer", desc);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPiZeroProducer);
