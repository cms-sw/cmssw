
#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithmNew_official.h"

#include <vector>

//--- Top-level of clustering algo

void TTClusterAlgorithmNew_official::clusterizeDetUnit(
    const edm::DetSet<Phase2TrackerDigi>& digis,
    TTClusterDetSetVec::FastFiller& clusters) const {

  if (digis.empty())
    return;
  
  if (enableClusterVetoes_ && isPSp_) {
    this->algoWithVetoes(digis, clusters);
  } else {
    this->algo(digis, clusters);
  }
  
}


//--- Simple clustering algo matching that in all FE electronics,
//--- except for case of PS-p sensors, where it misses the cluster vetoes.

void TTClusterAlgorithmNew_official::algo(
    const edm::DetSet<Phase2TrackerDigi>& digis,
    TTClusterDetSetVec::FastFiller& clusters) const {

  auto di = digis.begin();

  unsigned int widthCluster = 1;
  Phase2TrackerDigi firstDigi = *di;
  auto previous = firstDigi;

  ++di;

  for (; di != digis.end(); ++di) {
    auto digi = *di;

#ifdef VERIFY_PH2_TK_CLUS
    if (!(previous < digi))
      std::cout << "not ordered " << previous << ' ' << digi << std::endl;
#endif

    constexpr unsigned int rowsPerMPA = 120;

    const bool diffMPAchips = isPSp_ && (digi.row()/rowsPerMPA != previous.row()/rowsPerMPA);

    if (digi - previous == 1 && not diffMPAchips) {
      // Same column, adjacent row: extend the cluster (in r-phi).
      ++widthCluster;
      
    } else {
      
      // Finish the current cluster.
      
      if (widthCluster <= maxClusterWidth_) {
        clusters.push_back(TTCluster<Ref_Phase2TrackerDigi_>(detId_, upperSensor_, firstDigi, widthCluster));
      }

      // Start a new cluster.
      firstDigi = digi;
      widthCluster = 1;
    }

    previous = digi;
  }

  // Finish the final cluster.
  if (widthCluster <= maxClusterWidth_) {
    clusters.push_back(TTCluster<Ref_Phase2TrackerDigi_>(detId_, upperSensor_, firstDigi, widthCluster));
  }
}


//--- Complex clustering algo matching that in all FE electronics.
//--- It correctly describes the cluster vetoes of the PS-p sensors.
//--- (It also gives correct results for PS-s and 2S sensors,
//---  but for them, it is unnecessarily complex & CPU intensive.)

void TTClusterAlgorithmNew_official::algoWithVetoes(
    const edm::DetSet<Phase2TrackerDigi>& digis,
    TTClusterDetSetVec::FastFiller& clusters) const {
  
  struct BufferedCluster {
    TTCluster<Ref_Phase2TrackerDigi_> cluster;
    unsigned int twiceCentroid;
    bool veto = false;
  };

  struct BufferedColumn {
    std::vector<BufferedCluster> clusters;
    unsigned int ID = 99999; // number of the column
  };

  // These contain last three columns processed that contained clusters.
  BufferedColumn twoColumnsAgo;
  BufferedColumn previousColumn;
  BufferedColumn currentColumn;
  
  //--------------------------------------------------------------------
  
  auto makeBufferedCluster = [&](const Phase2TrackerDigi& firstDigi,
                                 unsigned int widthCluster) {
    unsigned int firstRow = firstDigi.row();

    // The centroid is:
    //
    //   firstRow + (widthCluster - 1) / 2
    //
    // Store twice the centroid so that half-integer centroids are
    // compared exactly, without floating point.
    unsigned int twiceCentroid = 2 * firstRow + widthCluster - 1;

    return BufferedCluster{
      TTCluster<Ref_Phase2TrackerDigi_>(detId_, upperSensor_, firstDigi, widthCluster),      
      twiceCentroid
    };
  };

  // Add clusters in a column to output collection
  
  auto outputColumn = [&](BufferedColumn& column) {
    for (const auto& b : column.clusters) {
      if (!b.veto)
        clusters.push_back(b.cluster);
    }
  };
  
  // Compare the last two columns with the new column.
  //
  // A pair of equal centroid rows means that the cluster in the
  // higher-numbered column is vetoed.
  //
  // Three equal centroid rows in consecutive columns means that
  // all three clusters are vetoed.
  auto processNewColumn = [&](BufferedColumn& oldest,
                              BufferedColumn& previous,
                              BufferedColumn& current) {

    // Cluster veto logic is only used by the MPA chip (PS-p sensors).
    //     b) If two PS-p clusters in neighbouring columns (r-z) have 
    //        the same row (r-phi) centroid, the one with higher column
    //         number is vetoed.
    //     c) If >= 3 PS-p clusters in neighbouring columns (r-z) have 
    //        the same row (r-phi) centroid, all these clusters are 
    //        vetoed.

    // Small complication -- PS-p modules have 32 columns, with 16
    // read out by MPA chips at each end of module. The veto conditions
    // only apply to columns read out from the same end.
 
    
    if (enableClusterVetoes_ && isPSp_) {

      constexpr unsigned int numColsPerEnd = 16;
      const bool currEnd = (current.ID  < numColsPerEnd);
      const bool prevEnd = (previous.ID < numColsPerEnd);
      const bool oldEnd  = (oldest.ID   < numColsPerEnd);
      
      // Identify clusters with same centroid position in neighbouring columns,
      // using fact that within a row, clusters are ordered by position,
      // so as to save CPU by not using a triple nested loop. 
      
      auto curr_clus = current.clusters.begin();
      auto prev_clus = previous.clusters.begin();
      auto old_clus  = oldest.clusters.begin();
      if (current.ID == previous.ID + 1 && currEnd == prevEnd) {
        // We have 2 neighbouring columns
        while (curr_clus != current.clusters.end() && prev_clus != previous.clusters.end()) {
          if (prev_clus->twiceCentroid < curr_clus->twiceCentroid) {
            prev_clus++;
          } else if (prev_clus->twiceCentroid > curr_clus->twiceCentroid) {
            curr_clus++;
          } else {
            // Two-column veto: cluster furthest from module end vetoed.
            if (currEnd) {
              curr_clus->veto = true;
            } else {
              prev_clus->veto = true;
            }
              
            if (current.ID == oldest.ID + 2 && currEnd == oldEnd) {
              // We have 3 neighbouring columns
              while (old_clus != oldest.clusters.end() &&
                     old_clus->twiceCentroid < curr_clus->twiceCentroid) old_clus++;
              
              if (old_clus != oldest.clusters.end() &&
                  old_clus->twiceCentroid == curr_clus->twiceCentroid) {
                // Three-column veto: all three clusters are vetoed.
                old_clus->veto = true;
                prev_clus->veto = true;
                // curr_clus already vetoed above.
              }
            }
            curr_clus++;
            prev_clus++;              
          }
        }
      }   
    }
    
    // -- OUTPUT CLUSTERS of oldest column, since no new column can veto them.
    outputColumn(oldest);
  };

  //--------------------------------------------------------------------
  
  auto di = digis.begin();

  unsigned int widthCluster = 1;
  Phase2TrackerDigi firstDigi = *di;
  auto previous = firstDigi;

  currentColumn.ID = firstDigi.column();

  ++di;

  for (; di != digis.end(); ++di) {
    auto digi = *di;

#ifdef VERIFY_PH2_TK_CLUS
    if (!(previous < digi))
      std::cout << "not ordered " << previous << ' ' << digi << std::endl;
#endif

    constexpr unsigned int rowsPerMPA = 120;

    const bool diffMPAchips = isPSp_ && (digi.row()/rowsPerMPA != previous.row()/rowsPerMPA);

    if (digi - previous == 1 && not diffMPAchips) {
      // Same column, adjacent row: extend the cluster (in r-phi).
      ++widthCluster;
      
    } else {
      
      // Finish the current cluster.
      
      if (widthCluster <= maxClusterWidth_) {
        currentColumn.clusters.push_back(makeBufferedCluster(firstDigi, widthCluster));
      }

      if (digi.column() != currentColumn.ID) {
        // A complete column has just been constructed.
        //
        // Once we have three columns, the oldest one can be
        // completely resolved.
        processNewColumn(twoColumnsAgo, previousColumn, currentColumn);

        twoColumnsAgo = std::move(previousColumn);
        previousColumn = std::move(currentColumn);
        
        currentColumn.ID = digi.column();
        currentColumn.clusters.clear();
      }

      // Start a new cluster.
      firstDigi = digi;
      widthCluster = 1;
    }

    previous = digi;
  }

  // Finish the final cluster.
  if (widthCluster <= maxClusterWidth_) {
    currentColumn.clusters.push_back(makeBufferedCluster(firstDigi, widthCluster));
  }

  // Since no new digis allowed There are no more columns, so we now trigger processing of a further column.

  processNewColumn(twoColumnsAgo, previousColumn, currentColumn);

  // -- OUTPUT CLUSTERS of final 2 columns, as not yet output,
  // -- and no further columns can veto them.

  outputColumn(previousColumn);  
  outputColumn(currentColumn);  
}
