/*
 * RecoTauCleaner
 *
 * Author: Evan K. Friis, UC Davis
 *
 * Given a input collection of PFTaus, produces an output collection of PFTaus
 * such that no two PFTaus come from the same PFJet.  If multiple taus in the
 * collection come from the same PFJet, (dirty) they are ordered according to a
 * list of cleaners.  Each cleaner is a RecoTauCleanerPlugin, and returns a
 * double corresponding to the 'quality' for a given tau - an example would be
 * the level of isolation.  The set of dirty taus is then ranked
 * lexicographically by these cleaners, and the best one is placed in the
 * output collection.
 *
 * $Id $
 */

#include <boost/ptr_container/ptr_vector.hpp>
#include <algorithm>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCleaningTools.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

class RecoTauCleaner : public edm::EDProducer {
  typedef reco::tau::RecoTauCleanerPlugin Cleaner;
  typedef boost::ptr_vector<Cleaner> CleanerList;

  // Predicate that determines if two taus 'overlap' i.e. share a base PFJet
  class RemoveDuplicateJets {
    public:
      bool operator()(const reco::PFTauRef& a, const reco::PFTauRef& b)
      { return a->jetRef() == b->jetRef(); }
  };

  public:
    explicit RecoTauCleaner(const edm::ParameterSet& pset);
    ~RecoTauCleaner() {}
    void produce(edm::Event& evt, const edm::EventSetup& es);

  private:
    // Define scoring predicate for taus
    typedef reco::tau::RecoTauLexicographicalRanking<
      CleanerList, reco::PFTauRef> Predicate;
    std::auto_ptr<Predicate> predicate_;
    edm::InputTag tauSrc_;
    CleanerList cleaners_;
};

RecoTauCleaner::RecoTauCleaner(const edm::ParameterSet& pset) {
  tauSrc_ = pset.getParameter<edm::InputTag>("tauSrc");
  // Build our list of quality plugins
  typedef std::vector<edm::ParameterSet> VPSet;
  // Get each of our tau builders
  const VPSet& cleaners = pset.getParameter<VPSet>("cleaners");
  for (VPSet::const_iterator cleanerPSet = cleaners.begin();
      cleanerPSet != cleaners.end(); ++cleanerPSet) {
    // Get plugin name
    const std::string& pluginType =
      cleanerPSet->getParameter<std::string>("plugin");
    // Build the plugin
    cleaners_.push_back(
        RecoTauCleanerPluginFactory::get()->create(pluginType, *cleanerPSet));
  }

  // Build the predicate that ranks our taus.  The predicate takes a list of
  // cleaners, and uses them to create a lexicographic ranking.
  predicate_ = std::auto_ptr<Predicate>(new Predicate(cleaners_));
  produces<reco::PFTauCollection>();
}

void RecoTauCleaner::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Update all our cleaners with the event info if they need it
  for (CleanerList::iterator cleaner = cleaners_.begin();
      cleaner != cleaners_.end(); ++cleaner) {
    cleaner->setup(evt, es);
  }

  // Get the input collection to clean
  edm::Handle<reco::PFTauCollection> pfTaus;
  evt.getByLabel(tauSrc_, pfTaus);

  // Build a local vector of Refs to the taus
  typedef std::vector<reco::PFTauRef> PFTauRefs;

  // Collection of all taus. Some are from the same PFJet. We must clean them.
  size_t nDirtyTaus = pfTaus->size();
  PFTauRefs dirty;
  dirty.reserve(nDirtyTaus);

  for (size_t iTau = 0; iTau < nDirtyTaus; ++iTau) {
    dirty.push_back(reco::PFTauRef(pfTaus, iTau));
  }

  // Sort the input tau refs according to our predicate
  std::sort(dirty.begin(), dirty.end(), *predicate_);

  // Clean the taus, ensuring that only one tau per jet is produced
  PFTauRefs cleanTaus = reco::tau::cleanOverlaps<PFTauRefs,
            RemoveDuplicateJets>(dirty);

  // create output collection
  std::auto_ptr<reco::PFTauCollection> output(new reco::PFTauCollection());
  output->reserve(cleanTaus.size());

  // std::cout << "*********      clean            **************" << std::endl;
  // Copy clean refs into output
  for (PFTauRefs::const_iterator tau = cleanTaus.begin();
       tau != cleanTaus.end(); ++tau) {
    output->push_back(**tau);
    // std::cout << std::setprecision(3) << **tau << " ";
    for (CleanerList::const_iterator cleaner = cleaners_.begin();
        cleaner != cleaners_.end(); ++cleaner) {
     // std::cout << cleaner->name() << ":"
     //   << std::setprecision(3) << (*cleaner)(*tau) << " ";
    }
    // std::cout << std::endl;
  }
  evt.put(output);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RecoTauCleaner);

