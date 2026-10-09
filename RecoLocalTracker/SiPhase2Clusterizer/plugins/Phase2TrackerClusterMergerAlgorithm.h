#ifndef RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterMergerAlgorithm_h
#define RecoLocalTracker_SiPhase2Clusterizer_Phase2TrackerClusterMergerAlgorithm_h

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

#include <algorithm>
#include <vector>

class Phase2TrackerClusterMergerAlgorithm {
public:
  inline void mergeDetUnit(const edmNew::DetSet<Phase2TrackerCluster1D>&,
                           Phase2TrackerCluster1DCollectionNew::FastFiller&,
                          const bool, const SiStripBadStrip*, const int) const;
};

void Phase2TrackerClusterMergerAlgorithm::mergeDetUnit(
    const edmNew::DetSet<Phase2TrackerCluster1D>& input,
    Phase2TrackerCluster1DCollectionNew::FastFiller& merged,
    const bool handleBadStrips_,
    const SiStripBadStrip* badStripPayload_,
    const int maxBadStrips
  ) const {
  if (input.empty())
    return;

  // The clusterizer emits its clusters ordered by (column, first row), but an upstream
  // producer applying a hardware cluster size cap need not, so order them explicitly.
  // Phase2TrackerCluster1D's own operator< compares the first strip while ignoring the
  // column, hence the comparator here.
  std::vector<Phase2TrackerCluster1D> ordered(input.begin(), input.end());
  std::sort(ordered.begin(), ordered.end(), [](const auto& one, const auto& other) {
    if (one.column() != other.column())
      return one.column() < other.column();
    return one.firstRow() < other.firstRow();
  });

  // Keeping the first constituent's digi rather than rebuilding one from (row, column)
  // preserves its over threshold bit, as the clusterizer itself does.
  auto ci = ordered.begin();
  Phase2TrackerDigi firstClusterDigi = ci->firstDigi();
  unsigned int sizeCluster = ci->size();
  unsigned int HIPbit = ci->threshold();
  ++ci;

  SiStripBadStrip::Range badStripObjects = badStripPayload_->getRange(input.detId());
  std::vector<unsigned int>::const_iterator badStripObject = badStripObjects.first;

  for (; ci != ordered.end(); ++ci) {
    
    Phase2TrackerDigi nextClusterDigi = ci->firstDigi();

    if (nextClusterDigi.column() == firstClusterDigi.column()) {
      
      // one past the last channel of the cluster
      const int firstClusterEdge = firstClusterDigi.channel() + sizeCluster;
      const int nextClusterFirst = nextClusterDigi.channel();

      if(firstClusterEdge == nextClusterFirst){
        sizeCluster += ci->size();
        HIPbit |= ci->threshold();
      }else if(handleBadStrips_ &&
        (firstClusterEdge + maxBadStrips >= nextClusterFirst)
      ){

        // merging clusters with bad strips between them
        bool mergeWithBadStrips = false;
        
        // iterate over bad strips to find if there's a bad strip block
        // between the two clusters
        for(; badStripObject != badStripObjects.second; ++badStripObject){
          
          const auto badStripObjectDecoded = badStripPayload_->decodePhase2(*badStripObject);
          // decoded to packed channel (same as digi)
          const auto badStripChannel = badStripObjectDecoded.firstStrip;
          const auto badStripBlockSize = badStripObjectDecoded.range;
          const auto badStripEdge = badStripChannel + badStripBlockSize;
          
          if(badStripEdge <= firstClusterEdge) continue;

          // strict matching, block of bad strips within the gap
          // no under/overflow
          if(badStripChannel == firstClusterEdge &&
            badStripEdge  == nextClusterFirst){
            mergeWithBadStrips = true;
            sizeCluster = sizeCluster + badStripBlockSize + ci->size();
            HIPbit |= ci->threshold();

            std::cout << "Merging cluster with edge " << firstClusterEdge << " to cluster with first strip " << nextClusterFirst << " separated by bad strip block with channel " << badStripChannel << " and size " << badStripBlockSize << std::endl;
          }           

          break;
          
        }
        
        if(!mergeWithBadStrips){
          merged.push_back(Phase2TrackerCluster1D(firstClusterDigi, sizeCluster, HIPbit));
          firstClusterDigi = nextClusterDigi;
          sizeCluster = ci->size();
          HIPbit = ci->threshold();               
        }

      }else{

        merged.push_back(Phase2TrackerCluster1D(firstClusterDigi, sizeCluster, HIPbit));
        firstClusterDigi = nextClusterDigi;
        sizeCluster = ci->size();
        HIPbit = ci->threshold();               

      }

    } else {
      merged.push_back(Phase2TrackerCluster1D(firstClusterDigi, sizeCluster, HIPbit));
      firstClusterDigi = ci->firstDigi();
      sizeCluster = ci->size();
      HIPbit = ci->threshold();
    }
  }
  merged.push_back(Phase2TrackerCluster1D(firstClusterDigi, sizeCluster, HIPbit));
}

#endif
