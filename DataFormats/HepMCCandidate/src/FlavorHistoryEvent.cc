#include "DataFormats/HepMCCandidate/interface/FlavorHistoryEvent.h"
#include "TLorentzVector.h"

#include <iostream>

using namespace reco;
using namespace std;

// Loop over flavor histories, count number of genjets with
// flavor b, c, or l
void FlavorHistoryEvent::cache() {
  bool verbose = false;

  if (verbose)
    cout << "----- Caching Flavor History Event  -----" << endl;
  // Set cached to false
  cached_ = false;
  nb_ = nc_ = 0;
  highestFlavor_ = 0;
  dR_ = 0.0;
  flavorSource_ = FlavorHistory::FLAVOR_NULL;

  // get list of flavor --> type --> dR.
  // Will sort this later to determine event classification
  vector<helpers::FlavorHistoryEventHelper> classification;

  // get iterators to the history vector
  const_iterator i = histories_.begin(), ibegin = histories_.begin(), iend = histories_.end();
  // loop over the history vector and count the number of
  // partons of flavors "b" and "c" that have a matched genjet.
  for (; i != iend; ++i) {
    FlavorHistory const& flavHist = *i;
    if (verbose)
      cout << "  Processing flavor history: " << i - ibegin << " = " << endl << flavHist << endl;
    CandidatePtr const& parton = flavHist.parton();
    flavor_type flavorSource = flavHist.flavorSource();

    // Now examine the matched jets to see what the classification should be.
    int pdgId = -1;
    if (parton.isNonnull())
      pdgId = abs(parton->pdgId());
    ShallowClonePtrCandidate const& matchedJet = flavHist.matchedJet();
    // Only count events with a matched genjet
    if (matchedJet.masterClonePtr().isNonnull()) {
      TLorentzVector p41(matchedJet.px(), matchedJet.py(), matchedJet.pz(), matchedJet.energy());
      if (pdgId == 5)
        nb_++;
      if (pdgId == 4)
        nc_++;

      // Get the sister genjet
      ShallowClonePtrCandidate const& sisterJet = i->sisterJet();
      TLorentzVector p42(sisterJet.px(), sisterJet.py(), sisterJet.pz(), sisterJet.energy());

      // Now check the source.
      double dR = -1;
      if (sisterJet.masterClonePtr().isNonnull()) {
        dR = p41.DeltaR(p42);
      }
      // Add to the vector to be sorted later
      if (verbose)
        cout << "Adding classification: pdgId = " << pdgId << ", flavorSource = " << flavorSource << ", dR = " << dR
             << endl;
      classification.push_back(helpers::FlavorHistoryEventHelper(pdgId, flavorSource, dR));
    } else {
      if (verbose)
        cout << "No matched jet found, not adding to classification list" << endl;
    }
  }

  // Sort by priority

  // Priority is:
  //
  //  1. flavor (5 > 4)
  //  2. type:
  //      2a. Flavor decay
  //      2b. Matrix element
  //      2c. Flavor excitation
  //      2d. Gluon splitting
  //  3. delta R (if applicable)
  if (!classification.empty()) {
    std::sort(classification.begin(), classification.end());

    if (verbose) {
      cout << "Writing out list of classifications" << endl;
      copy(classification.begin(), classification.end(), ostream_iterator<helpers::FlavorHistoryEventHelper>(cout, ""));
    }

    helpers::FlavorHistoryEventHelper const& best = *(classification.rbegin());
    dR_ = best.dR;
    highestFlavor_ = best.flavor;
    flavorSource_ = best.flavorSource;
  } else {
    dR_ = -1.0;
    highestFlavor_ = 0;
    flavorSource_ = FlavorHistory::FLAVOR_NULL;
  }

  // now we're cached, can return values quickly
  cached_ = true;
}
