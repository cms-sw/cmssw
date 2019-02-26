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
  minJetPt_ = ( pset.exists("minJetPt") ) ? pset.getParameter<double>("minJetPt") : -1.0;
  maxJetAbsEta_ = ( pset.exists("maxJetAbsEta") ) ? pset.getParameter<double>("maxJetAbsEta") : 99.0;

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

  // Check if we want to apply a final output selection
  if (pset.exists("outputSelection")) {
    std::string selection = pset.getParameter<std::string>("outputSelection");
    if (!selection.empty()) {
      outputSelector_.reset(
          new StringCutObjectSelector<reco::RecoTauPiZero>(selection));
    }
  }

  verbosity_ = ( pset.exists("verbosity") ) ?
    pset.getParameter<int>("verbosity") : 0;

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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauPiZeroProducer);
