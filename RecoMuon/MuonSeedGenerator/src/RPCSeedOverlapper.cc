/**
 *  See header file for a description of this class.
 *
 */

#include "RecoMuon/MuonSeedGenerator/src/RPCSeedOverlapper.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

using namespace std;
using namespace edm;

RPCSeedOverlapper::RPCSeedOverlapper() : rpcGeometry{nullptr} {
  isConfigured = false;
  isIOset = false;
}

RPCSeedOverlapper::~RPCSeedOverlapper() {}

void RPCSeedOverlapper::configure(const edm::ParameterSet &iConfig) {
  isCheckgoodOverlap = iConfig.getParameter<bool>("isCheckgoodOverlap");
  isCheckcandidateOverlap = iConfig.getParameter<bool>("isCheckcandidateOverlap");
  ShareRecHitsNumberThreshold = iConfig.getParameter<unsigned int>("ShareRecHitsNumberThreshold");
  isConfigured = true;
}

void RPCSeedOverlapper::setIO(std::vector<weightedTrajectorySeed> *goodweightedRef,
                              std::vector<weightedTrajectorySeed> *candidateweightedRef) {
  goodweightedSeedsRef = goodweightedRef;
  candidateweightedSeedsRef = candidateweightedRef;
  isIOset = true;
}

void RPCSeedOverlapper::unsetIO() { isIOset = false; }

void RPCSeedOverlapper::setGeometry(const RPCGeometry &iGeom) { rpcGeometry = &iGeom; }

void RPCSeedOverlapper::run() {
  if (isConfigured == false || isIOset == false || rpcGeometry == nullptr) {
    cout << "Configuration or IO is not set yet" << endl;
    return;
  }
  if (isCheckgoodOverlap == true)
    CheckOverlap(*rpcGeometry, goodweightedSeedsRef);
  if (isCheckcandidateOverlap == true)
    CheckOverlap(*rpcGeometry, candidateweightedSeedsRef);
}

void RPCSeedOverlapper::CheckOverlap(const RPCGeometry &rpcGeometry,
                                     std::vector<weightedTrajectorySeed> *weightedSeedsRef) {
  std::vector<weightedTrajectorySeed> sortweightedSeeds;
  std::vector<weightedTrajectorySeed> tempweightedSeeds;
  std::vector<TrackingRecHit const *> tempRecHits;

  while (!weightedSeedsRef->empty()) {
    cout << "Finding the weighted seeds group from " << weightedSeedsRef->size() << " seeds which share some recHits"
         << endl;
    // Take 1st seed in SeedsRef as referrence and find a collection which always share some recHits with some other
    tempRecHits.clear();
    tempweightedSeeds.clear();
    int N = 0;
    for (vector<weightedTrajectorySeed>::iterator itweightedseed = weightedSeedsRef->begin();
         itweightedseed != weightedSeedsRef->end();
         N++) {
      auto const &recHitsRange = itweightedseed->first.recHits();
      if (N == 0) {
        cout << "Always take the 1st weighted seed to be the referrence." << endl;
        for (auto const &hit : recHitsRange) {
          cout << "Put its recHits to tempRecHits" << endl;
          tempRecHits.push_back(&hit);
        }
        cout << "Put it to tempweightedSeeds" << endl;
        tempweightedSeeds.push_back(*itweightedseed);
        cout << "Then erase from weightedSeedsRef->" << endl;
        itweightedseed = weightedSeedsRef->erase(itweightedseed);
      } else {
        cout << "Come to other weighted seed for checking " << itweightedseed->first.nHits() << " recHits from "
             << tempRecHits.size() << " temp recHits" << endl;
        unsigned int ShareRecHitsNumber = 0;
        for (auto const &hit : recHitsRange) {
          if (isShareHit(tempRecHits, hit, rpcGeometry))
            ShareRecHitsNumber++;
        }
        if (ShareRecHitsNumber >= ShareRecHitsNumberThreshold) {
          cout << "This seed is found to belong to current share group" << endl;
          for (auto const &hit : recHitsRange) {
            if (!isShareHit(tempRecHits, hit, rpcGeometry)) {
              cout << "Put its extra recHits to tempRecHits" << endl;
              tempRecHits.push_back(&hit);
            }
          }
          cout << "Put it to tempSeeds" << endl;
          tempweightedSeeds.push_back(*itweightedseed);
          cout << "Then erase from SeedsRef" << endl;
          itweightedseed = weightedSeedsRef->erase(itweightedseed);
        } else
          itweightedseed++;
      }
    }
    // Find the best weighted seed and kick out those share recHits with it
    // The best weighted seed save in sortweightedSeeds, those don't share recHits with it will be push back to weightedSeedsRef for next while loop
    weightedTrajectorySeed bestweightedSeed;
    vector<weightedTrajectorySeed>::iterator bestweightediter;
    // Find the min Spt wrt Pt as the best Seed
    double Quality = 1000000;
    unsigned NumberofHits = 0;
    cout << "Find " << tempweightedSeeds.size() << " seeds into one trajectory group" << endl;
    for (vector<weightedTrajectorySeed>::iterator itweightedseed = tempweightedSeeds.begin();
         itweightedseed != tempweightedSeeds.end();
         itweightedseed++) {
      unsigned int nHits = itweightedseed->first.nHits();
      //std::vector<float> seed_error = itweightedseed->first.startingState().errorMatrix();
      //double Spt = seed_error[1];
      double weightedQuality = itweightedseed->second;
      cout << "Find a weighted seed with quality " << weightedQuality << endl;
      if ((NumberofHits < nHits) || (NumberofHits == nHits && weightedQuality < Quality)) {
        NumberofHits = nHits;
        Quality = weightedQuality;
        bestweightedSeed = *itweightedseed;
        bestweightediter = itweightedseed;
      }
    }
    cout << "Best good temp seed's quality is " << Quality << endl;
    sortweightedSeeds.push_back(bestweightedSeed);
    tempweightedSeeds.erase(bestweightediter);
    tempRecHits.clear();

    for (auto const &hit : bestweightedSeed.first.recHits()) {
      tempRecHits.push_back(&hit);
    }

    for (vector<weightedTrajectorySeed>::iterator itweightedseed = tempweightedSeeds.begin();
         itweightedseed != tempweightedSeeds.end();) {
      cout << "Checking the temp weighted seed's " << itweightedseed->first.nHits() << " hits to " << tempRecHits.size()
           << " temp recHits" << endl;
      bool isShare = false;
      for (auto const &hit : itweightedseed->first.recHits()) {
        if (isShareHit(tempRecHits, hit, rpcGeometry))
          isShare = true;
      }

      if (isShare == true) {
        cout << "Find one temp seed share some recHits with best weighted seed" << endl;
        itweightedseed = tempweightedSeeds.erase(itweightedseed);
      } else {
        cout << "This seed has no relation with best weighted seed" << endl;
        weightedSeedsRef->push_back(*itweightedseed);
        itweightedseed = tempweightedSeeds.erase(itweightedseed);
      }
    }
  }
  // At the end exchange SeedsRef with sortSeeds
  weightedSeedsRef->clear();
  *weightedSeedsRef = sortweightedSeeds;
}

bool RPCSeedOverlapper::isShareHit(const std::vector<TrackingRecHit const *> &recHits,
                                   const TrackingRecHit &hit,
                                   const RPCGeometry &rpcGeometry) {
  bool istheSame = false;
  unsigned int n = 1;
  cout << "Checking from " << recHits.size() << " temp recHits" << endl;

  LocalPoint lpos1 = hit.localPosition();
  DetId RPCId1 = hit.geographicalId();
  const GeomDetUnit *rpcroll1 = rpcGeometry.idToDetUnit(RPCId1);
  GlobalPoint gpos1 = rpcroll1->toGlobal(lpos1);
  cout << "The hit's position: " << gpos1.x() << ", " << gpos1.y() << ", " << gpos1.z() << endl;
  for (auto const &recHit : recHits) {
    cout << "Checking the " << (n++) << " th recHit from tempRecHits" << endl;
    LocalPoint lpos2 = recHit->localPosition();
    DetId RPCId2 = recHit->geographicalId();
    const GeomDetUnit *rpcroll2 = rpcGeometry.idToDetUnit(RPCId2);
    GlobalPoint gpos2 = rpcroll2->toGlobal(lpos2);
    cout << "The temp hit's position: " << gpos2.x() << ", " << gpos2.y() << ", " << gpos2.z() << endl;

    if ((gpos1.x() == gpos2.x()) && (gpos1.y() == gpos2.y()) && (gpos1.z() == gpos2.z())) {
      cout << "This hit is found to be the same" << endl;
      istheSame = true;
    }
  }
  return istheSame;
}
