#ifndef FASTSIMULATION_TRACKING_SEEDFINDER_H
#define FASTSIMULATION_TRACKING_SEEDFINDER_H

// system
#include <vector>
#include <functional>
#include <array>

// fastsim tracking
#include "FastSimulation/Tracking/interface/SeedingTree.h"
#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
#include "FastSimulation/Tracking/interface/SeedFinderSelector.h"

class TrackerTopology;
class FastTrackerRecHit;

class SeedFinder {
public:
private:
  std::vector<std::vector<SeedFinderSelector*> > _selectorFunctionsByHits;
  const SeedingTree<TrackingLayer>& _seedingTree;
  const TrackerTopology* _trackerTopology;

public:
  SeedFinder(const SeedingTree<TrackingLayer>& seedingTree, const TrackerTopology& trackerTopology)
      : _seedingTree(seedingTree), _trackerTopology(&trackerTopology) {}

  void addHitSelector(SeedFinderSelector* seedFinderSelector, unsigned int nHits) {
    if (_selectorFunctionsByHits.size() < nHits) {
      _selectorFunctionsByHits.resize(nHits);
      _selectorFunctionsByHits.reserve(nHits);
    }
    //shift indices by -1 so that _selectorFunctionsByHits[0] tests 1 hit
    _selectorFunctionsByHits[nHits - 1].push_back(seedFinderSelector);
  }

  std::vector<unsigned int> getSeed(const std::vector<const FastTrackerRecHit*>& trackerRecHits) const {
    std::vector<int> hitIndicesInTree(_seedingTree.numberOfNodes(), -1);
    //A SeedingNode is associated by its index to this list. The list stores the indices of the hits in 'trackerRecHits'
    /* example
           SeedingNode                     | hit index                 | hit
           -------------------------------------------------------------------------------
           index=  0:  [BPix1]             | hitIndicesInTree[0] (=1)  | trackerRecHits[1]
           index=  1:   -- [BPix2]         | hitIndicesInTree[1] (=3)  | trackerRecHits[3]
           index=  2:   --  -- [BPix3]     | hitIndicesInTree[2] (=4)  | trackerRecHits[4]
           index=  3:   --  -- [FPix1_pos] | hitIndicesInTree[3] (=6)  | trackerRecHits[6]
           index=  4:   --  -- [FPix1_neg] | hitIndicesInTree[4] (=7)  | trackerRecHits[7]

           The implementation has been chosen such that the tree only needs to be build once upon construction.
        */
    std::vector<TrajectorySeedHitCandidate> seedHitCandidates;
    for (const FastTrackerRecHit* trackerRecHit : trackerRecHits) {
      TrajectorySeedHitCandidate seedHitCandidate(trackerRecHit, _trackerTopology);
      seedHitCandidates.push_back(seedHitCandidate);
    }
    return iterateHits(0, seedHitCandidates, hitIndicesInTree, true);

    //TODO: create pairs of TrackingLayer -> remove TrajectorySeedHitCandidate class
  }

  //this method attempts to insert the hit at position 'trackerRecHit' in 'trackerRecHits'
  //into the seeding tree (stored as 'hitIndicesInTree')
  const SeedingNode<TrackingLayer>* insertHit(const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
                                              std::vector<int>& hitIndicesInTree,
                                              const SeedingNode<TrackingLayer>* node,
                                              unsigned int trackerHit) const {
    if (!node->getParent() || hitIndicesInTree[node->getParent()->getIndex()] >= 0) {
      if (hitIndicesInTree[node->getIndex()] < 0) {
        const TrajectorySeedHitCandidate& currentTrackerHit = trackerRecHits[trackerHit];
        if (currentTrackerHit.getTrackingLayer() != node->getData()) {
          return nullptr;
        }

        const unsigned int NHits = node->getDepth() + 1;
        if (_selectorFunctionsByHits.size() >= NHits) {
          //are there any selector functions stored for NHits?
          if (!_selectorFunctionsByHits[NHits - 1].empty()) {
            //fill vector of Hits from node to root to be passed to the selector function
            std::vector<const FastTrackerRecHit*> seedCandidateHitList(node->getDepth() + 1);
            seedCandidateHitList[node->getDepth()] = currentTrackerHit.hit();
            const SeedingNode<TrackingLayer>* parentNode = node->getParent();
            while (parentNode != nullptr) {
              seedCandidateHitList[parentNode->getDepth()] =
                  trackerRecHits[hitIndicesInTree[parentNode->getIndex()]].hit();
              parentNode = parentNode->getParent();
            }

            //loop over selector functions
            for (SeedFinderSelector* selectorFunction : _selectorFunctionsByHits[NHits - 1]) {
              if (!selectorFunction->pass(seedCandidateHitList)) {
                return nullptr;
              }
            }
          }
        }

        //the hit was not rejected by all selector functions -> insert it into the tree
        hitIndicesInTree[node->getIndex()] = trackerHit;
        if (node->getChildrenSize() == 0) {
          return node;
        }

        return nullptr;
      } else {
        for (unsigned int ichild = 0; ichild < node->getChildrenSize(); ++ichild) {
          const SeedingNode<TrackingLayer>* seed =
              insertHit(trackerRecHits, hitIndicesInTree, node->getChild(ichild), trackerHit);
          if (seed) {
            return seed;
          }
        }
      }
    }
    return nullptr;
  }

  std::vector<unsigned int> iterateHits(unsigned int start,
                                        const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
                                        std::vector<int> hitIndicesInTree,
                                        bool processSkippedHits) const {
    for (unsigned int irecHit = start; irecHit < trackerRecHits.size(); ++irecHit) {
      // only accept hits that are on one of the requested layers
      if (_seedingTree.getSingleSet().find(trackerRecHits[irecHit].getTrackingLayer()) ==
          _seedingTree.getSingleSet().end()) {
        continue;
      }

      unsigned int currentHitIndex = irecHit;

      for (unsigned int inext = currentHitIndex + 1; inext < trackerRecHits.size(); ++inext) {
        //if multiple hits are on the same layer -> follow all possibilities by recusion
        if (trackerRecHits[currentHitIndex].getTrackingLayer() == trackerRecHits[inext].getTrackingLayer()) {
          if (processSkippedHits) {
            //recusively call the method again with hit 'inext' but skip all following on the same layer though 'processSkippedHits=false'
            std::vector<unsigned int> seedHits = iterateHits(inext, trackerRecHits, hitIndicesInTree, false);
            if (!seedHits.empty()) {
              return seedHits;
            }
          }
          irecHit += 1;
        } else {
          break;
        }
      }

      //processSkippedHits=true

      const SeedingNode<TrackingLayer>* seedNode = nullptr;
      for (unsigned int iroot = 0; seedNode == nullptr && iroot < _seedingTree.numberOfRoots(); ++iroot) {
        seedNode = insertHit(trackerRecHits, hitIndicesInTree, _seedingTree.getRoot(iroot), currentHitIndex);
      }
      if (seedNode) {
        std::vector<unsigned int> seedIndices(seedNode->getDepth() + 1);
        while (seedNode) {
          seedIndices[seedNode->getDepth()] = hitIndicesInTree[seedNode->getIndex()];
          seedNode = seedNode->getParent();
        }
        return seedIndices;
      }
    }

    return std::vector<unsigned int>();
  }
};

#endif
