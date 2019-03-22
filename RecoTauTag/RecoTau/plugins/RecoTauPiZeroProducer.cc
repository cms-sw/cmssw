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

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_list.hpp>
#include <algorithm>
#include <functional>

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
    void print(const std::vector<reco::RecoTauPiZero>& piZeros,
               std::ostream& out);

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  private:
    typedef boost::ptr_vector<Builder> builderList;
    typedef boost::ptr_vector<Ranker> rankerList;
    typedef boost::ptr_vector<reco::RecoTauPiZero> PiZeroVector;
    typedef boost::ptr_list<reco::RecoTauPiZero> PiZeroList;

    typedef reco::tau::RecoTauLexicographicalRanking<rankerList,
            reco::RecoTauPiZero> PiZeroPredicate;

    builderList builders_;
    rankerList rankers_;
    std::auto_ptr<PiZeroPredicate> predicate_;
    double piZeroMass_;

    // Output selector
    std::auto_ptr<StringCutObjectSelector<reco::RecoTauPiZero> >
      outputSelector_;

    //consumes interface
    edm::EDGetTokenT<reco::CandidateView> cand_token;

    double minJetPt_;
    double maxJetAbsEta_;

    int verbosity_;
};

RecoTauPiZeroProducer::RecoTauPiZeroProducer(const edm::ParameterSet& pset) 
{
  cand_token = consumes<reco::CandidateView>( pset.getParameter<edm::InputTag>("jetSrc"));
  minJetPt_ = pset.getParameter<double>("minJetPt");
  maxJetAbsEta_ = pset.getParameter<double>("maxJetAbsEta");

  typedef std::vector<edm::ParameterSet> VPSet;
  // Get the mass hypothesis for the pizeros
  piZeroMass_ = pset.getParameter<double>("massHypothesis");

  // Get each of our PiZero builders
  const VPSet& builders = pset.getParameter<VPSet>("builders");

  for (VPSet::const_iterator builderPSet = builders.begin();
      builderPSet != builders.end(); ++builderPSet) {
    // Get plugin name
    const std::string& pluginType =
      builderPSet->getParameter<std::string>("plugin");
    // Build the plugin
    builders_.push_back(RecoTauPiZeroBuilderPluginFactory::get()->create(
          pluginType, *builderPSet, consumesCollector()));
  }

  // Get each of our quality rankers
  const VPSet& rankers = pset.getParameter<VPSet>("ranking");
  for (VPSet::const_iterator rankerPSet = rankers.begin();
      rankerPSet != rankers.end(); ++rankerPSet) {
    const std::string& pluginType =
      rankerPSet->getParameter<std::string>("plugin");
    rankers_.push_back(RecoTauPiZeroQualityPluginFactory::get()->create(
          pluginType, *rankerPSet));
  }

  // Build the sorting predicate
  predicate_ = std::auto_ptr<PiZeroPredicate>(new PiZeroPredicate(rankers_));

  // now all producers apply a final output selection
  std::string selection = pset.getParameter<std::string>("outputSelection");
  if (!selection.empty()) {
    outputSelector_.reset(
        new StringCutObjectSelector<reco::RecoTauPiZero>(selection));
  }

  verbosity_ = pset.getParameter<int>("verbosity");

  produces<reco::JetPiZeroAssociation>();
}

void RecoTauPiZeroProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  // Get a view of our jets via the base candidates
  edm::Handle<reco::CandidateView> jetView;
  evt.getByToken(cand_token, jetView);

  // Give each of our plugins a chance at doing something with the edm::Event
  for(auto& builder : builders_) {
    builder.setup(evt, es);
  }

  // Convert the view to a RefVector of actual PFJets
  reco::PFJetRefVector jetRefs =
      reco::tau::castView<reco::PFJetRefVector>(jetView);
  // Make our association
  std::unique_ptr<reco::JetPiZeroAssociation> association;

  if (!jetRefs.empty()) {
    edm::Handle<reco::PFJetCollection> pfJetCollectionHandle;
    evt.get(jetRefs.id(), pfJetCollectionHandle);
    association = std::make_unique<reco::JetPiZeroAssociation>(reco::PFJetRefProd(pfJetCollectionHandle));
  } else {
    association = std::make_unique<reco::JetPiZeroAssociation>();
  }

  // Loop over our jets
  for(auto const& jet : jetRefs) {

    if(jet->pt() - minJetPt_ < 1e-5) continue;
    if(std::abs(jet->eta()) - maxJetAbsEta_ > -1e-5) continue;
    // Build our global list of RecoTauPiZero
    PiZeroList dirtyPiZeros;

    // Compute the pi zeros from this jet for all the desired algorithms
    for(auto const& builder : builders_) {
      try {
        PiZeroVector result(builder(*jet));
        dirtyPiZeros.transfer(dirtyPiZeros.end(), result);
      } catch ( cms::Exception &exception) {
        edm::LogError("BuilderPluginException")
            << "Exception caught in builder plugin " << builder.name()
            << ", rethrowing" << std::endl;
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
      std::auto_ptr<reco::RecoTauPiZero> toAdd(
          dirtyPiZeros.pop_front().release());
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
        photonsInCleanCollection.insert(toAdd->daughterPtrVector().begin(),
                                        toAdd->daughterPtrVector().end());
        cleanPiZeros.push_back(*toAdd);
      } else {
        // Otherwise update the pizero that contains only the unique gammas and
        // add it back into the sorted list of dirty PiZeros
        toAdd->clearDaughters();
        // Add each of the unique daughters back to the pizero
        for(auto const& gamma : uniqueGammas) {
          toAdd->addDaughter(gamma);
        }
        // Update the four vector
        AddFourMomenta p4Builder_;
        p4Builder_.set(*toAdd);
        // Put this pi zero back into the collection of sorted dirty pizeros
        PiZeroList::iterator insertionPoint = std::lower_bound(
            dirtyPiZeros.begin(), dirtyPiZeros.end(), *toAdd, *predicate_);
        dirtyPiZeros.insert(insertionPoint, toAdd);
      }
    }
    // Apply the mass hypothesis if desired
    if (piZeroMass_ >= 0) {
      for( auto& cleanPiZero: cleanPiZeros )
         { cleanPiZero.setMass(this->piZeroMass_);};
    }
    // Add to association
    if ( verbosity_ >= 2 ) {
      print(cleanPiZeros, std::cout);
    }
    association->setValue(jet.key(), cleanPiZeros);
  }
  evt.put(std::move(association));
}

// Print some helpful information
void RecoTauPiZeroProducer::print(
    const std::vector<reco::RecoTauPiZero>& piZeros, std::ostream& out) {
  const unsigned int width = 25;
  for(auto const& piZero : piZeros) {
    out << piZero;
    out << "* Rankers:" << std::endl;
    for (rankerList::const_iterator ranker = rankers_.begin();
        ranker != rankers_.end(); ++ranker) {
      out << "* " << std::setiosflags(std::ios::left)
        << std::setw(width) << ranker->name()
        << " " << std::resetiosflags(std::ios::left)
        << std::setprecision(3) << (*ranker)(piZero);
      out << std::endl;
    }
  }
}

void
RecoTauPiZeroProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // common parameter descriptions
  edm::ParameterSetDescription vpsd_ranking;
  vpsd_ranking.add<std::string>("selectionPassFunction");
  vpsd_ranking.add<double>("selectionFailValue");
  vpsd_ranking.add<std::string>("selection");
  vpsd_ranking.add<std::string>("name");
  vpsd_ranking.add<std::string>("plugin");

  edm::ParameterSetDescription pset_signalQualityCuts;
  pset_signalQualityCuts.add<double>("maxDeltaZ", 0.4);
  pset_signalQualityCuts.add<double>("minTrackPt", 0.5);
  pset_signalQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  pset_signalQualityCuts.add<double>("maxTrackChi2", 100.0);
  pset_signalQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  pset_signalQualityCuts.add<double>("minGammaEt", 1.0);
  pset_signalQualityCuts.add<unsigned int>("minTrackHits", 3);
  pset_signalQualityCuts.addOptional<double>("minNeutralHadronEt");
  pset_signalQualityCuts.add<double>("maxTransverseImpactParameter", 0.1);
  pset_signalQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription pset_vxAssocQualityCuts;
  pset_vxAssocQualityCuts.add<double>("minTrackPt", 0.5);
  pset_vxAssocQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  pset_vxAssocQualityCuts.add<double>("maxTrackChi2", 100.0);
  pset_vxAssocQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  pset_vxAssocQualityCuts.add<double>("minGammaEt", 1.0);
  pset_vxAssocQualityCuts.add<unsigned int>("minTrackHits", 3);
  pset_vxAssocQualityCuts.add<double>("maxTransverseImpactParameter", 0.1);
  pset_vxAssocQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription pset_isolationQualityCuts;
  pset_isolationQualityCuts.add<double>("maxDeltaZ", 0.2);
  pset_isolationQualityCuts.add<double>("minTrackPt", 1.0);
  pset_isolationQualityCuts.add<double>("minTrackVertexWeight", -1.0);
  pset_isolationQualityCuts.add<double>("maxTrackChi2", 100.0);
  pset_isolationQualityCuts.add<unsigned int>("minTrackPixelHits", 0);
  pset_isolationQualityCuts.add<double>("minGammaEt", 1.5);
  pset_isolationQualityCuts.add<unsigned int>("minTrackHits", 8);
  pset_isolationQualityCuts.add<double>("maxTransverseImpactParameter", 0.03);
  pset_isolationQualityCuts.addOptional<bool>("useTracksInsteadOfPFHadrons");

  edm::ParameterSetDescription pset_qualityCuts;
  pset_qualityCuts.add<edm::ParameterSetDescription>("signalQualityCuts",    pset_signalQualityCuts);
  pset_qualityCuts.add<edm::ParameterSetDescription>("vxAssocQualityCuts",   pset_vxAssocQualityCuts);
  pset_qualityCuts.add<edm::ParameterSetDescription>("isolationQualityCuts", pset_isolationQualityCuts);
  pset_qualityCuts.add<std::string>("leadingTrkOrPFCandOption", "leadPFCand");
  pset_qualityCuts.add<std::string>("pvFindingAlgo", "closestInDeltaZ");
  pset_qualityCuts.add<edm::InputTag>("primaryVertexSrc", edm::InputTag("offlinePrimaryVertices"));
  pset_qualityCuts.add<bool>("vertexTrackFiltering", false);
  pset_qualityCuts.add<bool>("recoverLeadingTrk", false);


  {
    // ak4PFJetsLegacyTaNCPiZeros
    edm::ParameterSetDescription desc;
    desc.add<double>("massHypothesis", 0.136);
    desc.addVPSet("ranking", vpsd_ranking);
    desc.add<int>("verbosity", 0);
    desc.add<double>("maxJetAbsEta", 2.5);
    desc.add<std::string>("outputSelection", "pt > 1.5");
    desc.add<double>("minJetPt", 14.0);
    desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJets"));

    {
      edm::ParameterSetDescription vpsd_builders;
      vpsd_builders.add<edm::ParameterSetDescription>("qualityCuts", pset_qualityCuts);
      vpsd_builders.add<std::string>("name", "1");
      vpsd_builders.add<std::string>("plugin", "RecoTauPiZeroTrivialPlugin");
      vpsd_builders.add<int>("verbosity", 0);

      vpsd_builders.addOptional<bool>("makeCombinatoricStrips");
      vpsd_builders.addOptional<int>("maxStripBuildIterations");
      vpsd_builders.addOptional<double>("minGammaEtStripAdd");
      vpsd_builders.addOptional<double>("minGammaEtStripSeed");
      vpsd_builders.addOptional<double>("minStripEt");
      vpsd_builders.addOptional<std::vector<int>>("stripCandidatesParticleIds");
      vpsd_builders.addOptional<bool>("updateStripAfterEachDaughter");
      vpsd_builders.addOptional<bool>("applyElecTrackQcuts");

      desc.addVPSet("builders", vpsd_builders);
    }

    descriptions.add("ak4PFJetsLegacyTaNCPiZeros", desc);
  }

  {
    // ak4PFJetsRecoTauGreedyPiZeros
    edm::ParameterSetDescription desc;
    desc.add<double>("massHypothesis", 0.136);
    desc.addVPSet("ranking", vpsd_ranking);
    desc.add<int>("verbosity", 0);
    desc.add<double>("maxJetAbsEta", 2.5);
    desc.add<std::string>("outputSelection", "pt > 1.5");
    desc.add<double>("minJetPt", 14.0);
    desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJets"));
    {
      edm::ParameterSetDescription vpsd_builders;
      vpsd_builders.add<edm::ParameterSetDescription>("qualityCuts", pset_qualityCuts);
      vpsd_builders.add<int>("maxInputStrips", 5);
      vpsd_builders.add<std::string>("name", "cs");
      vpsd_builders.add<std::string>("plugin", "RecoTauPiZeroStripPlugin");
      vpsd_builders.add<double>("stripMassWhenCombining", 0.0);
      vpsd_builders.add<double>("stripPhiAssociationDistance", 0.2);
      vpsd_builders.add<double>("stripEtaAssociationDistance", 0.05);
      vpsd_builders.add<int>("verbosity", 0);

      vpsd_builders.addOptional<bool>("makeCombinatoricStrips");
      vpsd_builders.addOptional<int>("maxStripBuildIterations");
      vpsd_builders.addOptional<double>("minGammaEtStripAdd");
      vpsd_builders.addOptional<double>("minGammaEtStripSeed");
      vpsd_builders.addOptional<double>("minStripEt");
      vpsd_builders.addOptional<std::vector<int>>("stripCandidatesParticleIds");
      vpsd_builders.addOptional<bool>("updateStripAfterEachDaughter");
      vpsd_builders.addOptional<bool>("applyElecTrackQcuts");

      desc.addVPSet("builders", vpsd_builders);
    }
    descriptions.add("ak4PFJetsRecoTauGreedyPiZeros", desc);
  }

  {
    // ak4PFJetsRecoTauPiZeros
    edm::ParameterSetDescription desc;
    desc.add<double>("massHypothesis", 0.136);
    desc.addVPSet("ranking", vpsd_ranking);
    desc.add<int>("verbosity", 0);
    desc.add<double>("maxJetAbsEta", 2.5);
    desc.add<std::string>("outputSelection", "pt > 1.5");
    desc.add<double>("minJetPt", 14.0);
    desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJets"));
    {
      edm::ParameterSetDescription vpsd_builders;

      vpsd_builders.add<edm::ParameterSetDescription>("qualityCuts", pset_qualityCuts);
      vpsd_builders.add<std::string>("name", "2");
      vpsd_builders.add<std::string>("plugin", "RecoTauPiZeroCombinatoricPlugin");
      vpsd_builders.add<double>("maxMass", -1.0);
      vpsd_builders.add<double>("minMass", 0.0);
      vpsd_builders.add<unsigned int>("choose", 2);
      vpsd_builders.addOptional<unsigned int>("maxInputGammas");
      vpsd_builders.add<int>("verbosity", 0);

      vpsd_builders.addOptional<bool>("makeCombinatoricStrips");
      vpsd_builders.addOptional<int>("maxStripBuildIterations");
      vpsd_builders.addOptional<double>("minGammaEtStripAdd");
      vpsd_builders.addOptional<double>("minGammaEtStripSeed");
      vpsd_builders.addOptional<double>("minStripEt");
      vpsd_builders.addOptional<std::vector<int>>("stripCandidatesParticleIds");
      vpsd_builders.addOptional<bool>("updateStripAfterEachDaughter");
      vpsd_builders.addOptional<bool>("applyElecTrackQcuts");
      {
       	edm::ParameterSetDescription psd0;
        psd0.add<std::string>("function", "TMath::Min(0.3, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))");
        psd0.add<double>("par1", 0.707716);
        psd0.add<double>("par0", 0.352476);
        vpsd_builders.addOptional<edm::ParameterSetDescription>("stripPhiAssociationDistance", psd0);
      }
      {
       	edm::ParameterSetDescription psd0;
        psd0.add<std::string>("function", "TMath::Min(0.15, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))");
        psd0.add<double>("par1", 0.658701);
        psd0.add<double>("par0", 0.197077);
        vpsd_builders.addOptional<edm::ParameterSetDescription>("stripEtaAssociationDistance", psd0);
      }

      desc.addVPSet("builders", vpsd_builders);
    }

    descriptions.add("ak4PFJetsRecoTauPiZeros", desc);
  }

  {
    // ak4PFJetsLegacyHPSPiZeros
    edm::ParameterSetDescription desc;
    desc.add<double>("massHypothesis", 0.136);
    desc.addVPSet("ranking", vpsd_ranking);
    desc.add<int>("verbosity", 0);
    desc.add<double>("maxJetAbsEta", 2.5);
    desc.add<std::string>("outputSelection", "pt > 0");
    desc.add<double>("minJetPt", 14.0);
    desc.add<edm::InputTag>("jetSrc", edm::InputTag("ak4PFJets"));
    {
      edm::ParameterSetDescription vpsd_builders;
      vpsd_builders.setAllowAnything();  //This is done because due to the modification in https://github.com/cms-sw/cmssw/blob/master/RecoTauTag/RecoTau/python/RecoTauPiZeroProducer_cfi.py#L18-L25
      // both of the following uncommented version need to be accepted.
      //{
      //  edm::ParameterSetDescription psd0;
      //  psd0.add<std::string>("function", "TMath::Min(0.3, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))");
      //  psd0.add<double>("par1", 0.707716);
      //  psd0.add<double>("par0", 0.352476);
      //  vpsd_builders.addOptional<edm::ParameterSetDescription>("stripPhiAssociationDistance", psd0);
      //}
      //{
      //  edm::ParameterSetDescription psd0;
      //  psd0.add<std::string>("function", "TMath::Min(0.15, TMath::Max(0.05, [0]*TMath::Power(pT, -[1])))");
      //  psd0.add<double>("par1", 0.658701);
      //  psd0.add<double>("par0", 0.197077);
      //  vpsd_builders.addOptional<edm::ParameterSetDescription>("stripEtaAssociationDistance", psd0);
      //}
      //vpsd_builders.addOptional<double>("stripPhiAssociationDistance", 0.2);
      //vpsd_builders.addOptional<double>("stripEtaAssociationDistance", 0.05);
      vpsd_builders.add<edm::ParameterSetDescription>("qualityCuts", pset_qualityCuts);

      vpsd_builders.add<std::string>("name");
      vpsd_builders.add<std::string>("plugin");
      vpsd_builders.add<int>("verbosity", 0);

      vpsd_builders.addOptional<bool>("makeCombinatoricStrips");
      vpsd_builders.addOptional<int>("maxStripBuildIterations");
      vpsd_builders.addOptional<double>("minGammaEtStripAdd");
      vpsd_builders.addOptional<double>("minGammaEtStripSeed");
      vpsd_builders.addOptional<double>("minStripEt");
      vpsd_builders.addOptional<std::vector<int>>("stripCandidatesParticleIds");
      vpsd_builders.addOptional<bool>("updateStripAfterEachDaughter");
      vpsd_builders.addOptional<bool>("applyElecTrackQcuts");

      desc.addVPSet("builders", vpsd_builders);
    }

    descriptions.add("ak4PFJetsLegacyHPSPiZeros", desc);
    descriptions.add("ak4PFJetsLegacyHPSPiZerosBoosted", desc); // this one is generated in configs with a strange procedure
    descriptions.add("pfJetsLegacyHPSPiZeros", desc);
    // RecoTauTag/Configuration/python/boostedHPSPFTaus_cfi.py
    //    process.PATTauSequenceBoosted = cloneProcessingSnippet(process,process.PATTauSequence, "Boosted", addToTask = True)
  }

  {
    //hltPFTauPiZeros & hltPFTauPiZerosReg
    edm::ParameterSetDescription desc;
    desc.add<double>("massHypothesis", 0.136);
    desc.addVPSet("ranking", vpsd_ranking);
    desc.add<int>("verbosity", 0);
    desc.add<double>("maxJetAbsEta", 99.0);
    desc.add<std::string>("outputSelection", "pt > 0");
    desc.add<double>("minJetPt", -1.0);
    desc.add<edm::InputTag>("jetSrc");

    {
      edm::ParameterSetDescription vpsd_builders;
      {
        edm::ParameterSetDescription pset_hlt_qualityCuts;
        pset_hlt_qualityCuts.add<edm::ParameterSetDescription>("signalQualityCuts", pset_signalQualityCuts);
        pset_hlt_qualityCuts.add<std::string>("pvFindingAlgo", "closestInDeltaZ");
        pset_hlt_qualityCuts.add<bool>("vertexTrackFiltering", false);
        pset_hlt_qualityCuts.add<edm::InputTag>("primaryVertexSrc", edm::InputTag("hltPixelVertices"));
        pset_hlt_qualityCuts.add<bool>("recoverLeadingTrk", false);
        vpsd_builders.add<edm::ParameterSetDescription>("qualityCuts", pset_hlt_qualityCuts);
      }
      vpsd_builders.add<std::string>("name", "s");
      vpsd_builders.add<std::string>("plugin", "RecoTauPiZeroStripPlugin2");
      vpsd_builders.add<double>("stripPhiAssociationDistance", 0.2);
      vpsd_builders.add<double>("stripEtaAssociationDistance", 0.05);
      vpsd_builders.add<int>("verbosity", 0);

      vpsd_builders.addOptional<bool>("makeCombinatoricStrips");
      vpsd_builders.addOptional<int>("maxStripBuildIterations");
      vpsd_builders.addOptional<double>("minGammaEtStripAdd");
      vpsd_builders.addOptional<double>("minGammaEtStripSeed");
      vpsd_builders.addOptional<double>("minStripEt");
      vpsd_builders.addOptional<std::vector<int>>("stripCandidatesParticleIds");
      vpsd_builders.addOptional<bool>("updateStripAfterEachDaughter");
      vpsd_builders.addOptional<bool>("applyElecTrackQcuts");

      desc.addVPSet("builders", vpsd_builders);
    }
    descriptions.add("hltPFTauPiZeros", desc);
    descriptions.add("hltPFTauPiZerosReg", desc);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPiZeroProducer);
