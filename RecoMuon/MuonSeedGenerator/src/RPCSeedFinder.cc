/**
 *  See header file for a description of this class.
 *
 */

#include "RecoMuon/MuonSeedGenerator/src/RPCSeedFinder.h"
#include <iomanip>

using namespace std;
using namespace edm;

RPCSeedFinder::RPCSeedFinder() {
  // Initiate the member
  isrecHitsset = false;
  isConfigured = false;
  isOutputset = false;
  isEventSetupset = false;
  oneSeed.clear();
}

RPCSeedFinder::~RPCSeedFinder() {}

void RPCSeedFinder::configure(const edm::ParameterSet &iConfig) {
  oneSeed.configure(iConfig);
  isConfigured = true;
}

void RPCSeedFinder::setOutput(std::vector<weightedTrajectorySeed> *goodweightedRef,
                              std::vector<weightedTrajectorySeed> *candidateweightedRef) {
  goodweightedSeedsRef = goodweightedRef;
  candidateweightedSeedsRef = candidateweightedRef;
  isOutputset = true;
}

void RPCSeedFinder::setrecHits(ConstMuonRecHitContainer &recHits) {
  oneSeed.clear();
  for (ConstMuonRecHitContainer::const_iterator iter = recHits.begin(); iter != recHits.end(); iter++)
    oneSeed.add(*iter);
  isrecHitsset = true;
}

void RPCSeedFinder::setEventSetup(const MagneticField &field, const RPCGeometry &rpcGeom) {
  pField = &field;
  pRPCGeom = &rpcGeom;
  isEventSetupset = true;
}

void RPCSeedFinder::seed() {
  cout << "[RPCSeedFinder] --> seeds called" << endl;

  if (isrecHitsset == false || isOutputset == false || isConfigured == false || isEventSetupset == false) {
    cout << "Configuration or IO is not set yet" << endl;
    return;
  }

  weightedTrajectorySeed theweightedSeed;
  int isGoodSeed = 0;

  theweightedSeed = oneSeed.seed(*pField, *pRPCGeom, isGoodSeed);
  // Push back the good seed
  if (isGoodSeed == 1) {
    cout << "[RPCSeedFinder] --> Seeds from " << oneSeed.nrhit() << " recHits." << endl;
    goodweightedSeedsRef->push_back(theweightedSeed);
  }
  // Push back cadidate seed but not the fake seed
  if (isGoodSeed >= 0) {
    candidateweightedSeedsRef->push_back(theweightedSeed);
  }

  // Unset the signal
  oneSeed.clear();
  isrecHitsset = false;
}
