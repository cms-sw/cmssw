#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/Common/interface/Association.h"
//#include "DataFormats/Common/interface/AssociativeIterator.h"

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/iterator/filter_iterator.hpp>

namespace tautools {

class RecoTauDistanceFromTruthPlugin : public reco::tau::RecoTauCleanerPlugin {
  public:
    RecoTauDistanceFromTruthPlugin(const edm::ParameterSet& pset);
    virtual ~RecoTauDistanceFromTruthPlugin() {}
    double operator()(const reco::PFTauRef&) const;
    void beginEvent();
  private:
    edm::InputTag matchingSrc_;
    typedef edm::Association<reco::GenJetCollection> GenJetAssociation;
    edm::Handle<GenJetAssociation> genTauMatch_;
};

RecoTauDistanceFromTruthPlugin::RecoTauDistanceFromTruthPlugin(
    const edm::ParameterSet& pset): reco::tau::RecoTauCleanerPlugin(pset) {
  matchingSrc_ = pset.getParameter<edm::InputTag>("matching");
}

void RecoTauDistanceFromTruthPlugin::beginEvent() {
  // Load the matching information
  evt()->getByLabel(matchingSrc_, genTauMatch_);
}

// Helpers
namespace {
  // Returns the squared momentum difference between two candidates
  double momentumDifference(const reco::Candidate* candA,
      const reco::Candidate* candB) {
    reco::Candidate::LorentzVector difference =
      candA->p4() - candB->p4();
    return difference.P2();
  }

  // Finds the best match for an input <cand> from an input colleciton.
  // Only objects with matching charge are considered.  The best match
  // has the lowest [momentumDifference] with the input <cand>
  template<typename InputIterator>
  InputIterator findBestMatch(const reco::Candidate* cand,
      InputIterator begin, InputIterator end) {

    typedef const reco::Candidate* CandPtr;
    using boost::bind;
    using boost::function;
    using boost::filter_iterator;
    // Build a charge matching function
    typedef function<bool (CandPtr)> CandPtrBoolFn;
    CandPtrBoolFn chargeMatcher =
      bind(&reco::Candidate::charge, cand) == bind(&reco::Candidate::charge, _1);

    // Only match those objects that have the same charge
    typedef filter_iterator<CandPtrBoolFn, InputIterator> Iterator;
    Iterator begin_filtered(chargeMatcher, begin, end);
    Iterator end_filtered(chargeMatcher, end, end);

    Iterator result = std::min_element(begin_filtered, end_filtered,
        momentumDifference);
    return result.base();
  }
} // end anon. namespace

double RecoTauDistanceFromTruthPlugin::operator()(const reco::PFTauRef& tauRef) const {

  GenJetAssociation::reference_type truth = (*genTauMatch_)[tauRef];

  // Check if the matching exists, if not return +infinity
  if (truth.isNull())
    return std::numeric_limits<double>::infinity();

  // screw all this noise
  return std::abs(tauRef->pt() - truth->pt());

  typedef const reco::Candidate* CandPtr;
  typedef std::set<CandPtr> CandPtrSet;
  typedef std::vector<CandPtr> CandPtrVector;
  typedef std::list<CandPtr> CandPtrList;
  // Store all of our reco and truth candidates
  CandPtrList recoCands;
  CandPtrSet truthCandSet;

  BOOST_FOREACH(const reco::RecoTauPiZero& piZero,
      tauRef->signalPiZeroCandidates()) {
    recoCands.push_back(&piZero);
  }

  BOOST_FOREACH(const reco::PFCandidateRef& pfch,
      tauRef->signalPFChargedHadrCands()) {
    recoCands.push_back(pfch.get());
  }

  // Use a set to contain the truth pointers so we ensure that no pizeros
  // are entered twice.
  BOOST_FOREACH(const reco::CandidatePtr& ptr,
      truth->daughterPtrVector()) {
    // Get mother pi zeros in the case of gammas
    if (ptr->pdgId() == 22)
      truthCandSet.insert(ptr->mother());
    else
      truthCandSet.insert(ptr.get());
  }

  //Convert truth cands from set to vector so we can sort it.
  CandPtrVector truthCands(truthCandSet.begin(), truthCandSet.end());

  // Sort the truth candidates by descending pt
  std::sort(truthCands.begin(), truthCands.end(),
      boost::bind(&reco::Candidate::pt, _1) > boost::bind(&reco::Candidate::pt, _2));

  double quality = 0.0;
  BOOST_FOREACH(CandPtr truthCand, truthCands) {
    // Find the best reco match for this truth cand
    CandPtrList::iterator recoMatch = findBestMatch(truthCand,
        recoCands.begin(), recoCands.end());

    // Check if this truth cand is matched
    if (recoMatch != recoCands.end()) {
      // Add a penalty factor based on how different the reconstructed
      // is w.r.t. the true momentum
      quality += momentumDifference(truthCand, *recoMatch);
      // Remove this reco cand from future matches
      recoCands.erase(recoMatch);
    } else {
      // this truth cand was not matched!
      quality += truthCand->p4().P2();
    }
  }

  // Now add a penalty for the unmatched reco stuff
  BOOST_FOREACH(CandPtr recoCand, recoCands) {
    quality += recoCand->p4().P2();
  }

  return quality;
}

} // end tautools namespace


// Register our plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauCleanerPluginFactory, tautools::RecoTauDistanceFromTruthPlugin, "RecoTauDistanceFromTruthPlugin");
