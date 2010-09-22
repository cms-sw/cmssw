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
 * $Id $
 */ 

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCleaningTools.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/JetPiZeroAssociation.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/foreach.hpp>

#include <algorithm>


namespace {
  // Class that checks if two PiZeros contain any of the same daughters 
  class PiZeroOverlapChecker {
    public:
      bool operator()(const reco::RecoTauPiZero& a, 
          const reco::RecoTauPiZero& b) {
        typedef std::vector<reco::CandidatePtr> daughters;
        daughters aDaughters = a.daughterPtrVector();
        daughters bDaughters = b.daughterPtrVector();
        // Check if they share any daughters
        daughters::const_iterator result = std::find_first_of(
            aDaughters.begin(), aDaughters.end(), 
            bDaughters.begin(), bDaughters.end());
        return(result != aDaughters.end());
      }
  };
}

using namespace reco;

class RecoTauPiZeroProducer : public edm::EDProducer {
  public:
    typedef reco::tau::RecoTauPiZeroBuilderPlugin Builder;
    typedef reco::tau::RecoTauPiZeroQualityPlugin Ranker;

    explicit RecoTauPiZeroProducer(const edm::ParameterSet& pset);
    ~RecoTauPiZeroProducer(){};
    void produce(edm::Event& evt, const edm::EventSetup& es);
    void print(const std::vector<RecoTauPiZero>& piZeros, std::ostream& out);

  private:
    typedef boost::ptr_vector<Builder> builderList;
    typedef boost::ptr_vector<Ranker> rankerList;
    typedef reco::tau::RecoTauLexicographicalRanking<rankerList, RecoTauPiZero> 
      PiZeroPredicate;
    edm::InputTag jetSrc_;
    builderList builders_;
    rankerList rankers_;
    std::auto_ptr<PiZeroPredicate> predicate_;
};

RecoTauPiZeroProducer::RecoTauPiZeroProducer(const edm::ParameterSet& pset) {
  jetSrc_ = pset.getParameter<edm::InputTag>("jetSrc");

  typedef std::vector<edm::ParameterSet> VPSet;
  // Get each of our PiZero builders
  const VPSet& builders = pset.getParameter<VPSet>("builders");

  for(VPSet::const_iterator builderPSet = builders.begin(); 
      builderPSet != builders.end(); ++builderPSet) {
    // Get plugin name
    const std::string& pluginType = 
      builderPSet->getParameter<std::string>("plugin");
    // Build the plugin
    builders_.push_back(RecoTauPiZeroBuilderPluginFactory::get()->create(
          pluginType, *builderPSet));
  }

  // Get each of our quality rankers
  const VPSet& rankers = pset.getParameter<VPSet>("ranking"); 
  for(VPSet::const_iterator rankerPSet = rankers.begin(); 
      rankerPSet != rankers.end(); ++rankerPSet) {
    const std::string& pluginType = 
      rankerPSet->getParameter<std::string>("plugin");
    rankers_.push_back(RecoTauPiZeroQualityPluginFactory::get()->create(
          pluginType, *rankerPSet));
  }

  // Build the sorting predicate
  predicate_ = std::auto_ptr<PiZeroPredicate>(new PiZeroPredicate(rankers_));

  produces<JetPiZeroAssociation>();
}

void RecoTauPiZeroProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<PFJetCollection> pfJets;
  evt.getByLabel(jetSrc_, pfJets);

  // Make our association
  std::auto_ptr<JetPiZeroAssociation> association(
      new JetPiZeroAssociation(PFJetRefProd(pfJets)));

  size_t iJet = 0;
  // Loop over our jets
  for(PFJetCollection::const_iterator jet = pfJets->begin(); 
      jet != pfJets->end(); ++jet, ++iJet) {
    typedef std::vector<RecoTauPiZero> PiZeroVector;
    // Build our global list of RecoTauPiZero
    PiZeroVector dirtyPiZeros;

    // Compute the pi zeros from this jet for all the desired algorithms
    BOOST_FOREACH(const Builder& builder, builders_) {
      PiZeroVector result(builder(*jet));
      dirtyPiZeros.insert(dirtyPiZeros.end(), result.begin(), result.end());
    }

    //td::cout << "BEFORE SORTING" << std::endl;
    //print(dirtyPiZeros, std::cout);

    // Rank the candidates according to our quality plugins
    std::sort(dirtyPiZeros.begin(), dirtyPiZeros.end(), *predicate_);

    //std::cout << "APRES SORTING" << std::endl;
    //print(dirtyPiZeros, std::cout);

    // Now clean the list to ensure that no photon is counted twice
    PiZeroVector cleanPiZeros = reco::tau::cleanOverlaps<PiZeroVector, 
                 PiZeroOverlapChecker>(dirtyPiZeros);

    // Sort the clean pizeros by pt
    std::sort(cleanPiZeros.begin(), cleanPiZeros.end(), 
        reco::tau::SortByDescendingPt<RecoTauPiZero>());

    //std::cout << "CLEANED" << std::endl;
    //print(cleanPiZeros, std::cout);

    // Add to association
    association->setValue(iJet, cleanPiZeros);
  }
  evt.put(association);
}

// Print some helpful information
void RecoTauPiZeroProducer::print(
    const std::vector<reco::RecoTauPiZero>& piZeros, std::ostream& out) {
  const unsigned int width = 25;
  BOOST_FOREACH(const reco::RecoTauPiZero& piZero, piZeros) {
    out << piZero;
    out << "* Rankers:" << std::endl;
    for(rankerList::const_iterator ranker = rankers_.begin(); 
        ranker != rankers_.end(); ++ranker)
    {
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

