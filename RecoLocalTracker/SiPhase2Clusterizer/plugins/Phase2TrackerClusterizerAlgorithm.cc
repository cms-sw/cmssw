#include "RecoLocalTracker/SiPhase2Clusterizer/interface/Phase2TrackerClusterizerAlgorithm.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"

/*
 * Initialise the clusterizer algorithm
 */

Phase2TrackerClusterizerAlgorithm::Phase2TrackerClusterizerAlgorithm(unsigned int maxClusterSize, unsigned int maxNumberClusters) : maxClusterSize_(maxClusterSize), maxNumberClusters_(maxNumberClusters), nrows_(0), ncols_(0) { }

/* 
 * Change the size of the 2D matrix for this module (varies from pixel to strip modules)
 */

void Phase2TrackerClusterizerAlgorithm::setup(const PixelGeomDetUnit* pixDet) {
    const PixelTopology& topol(pixDet->specificTopology());
    nrows_ = topol.nrows();
    ncols_ = topol.ncolumns();
    matrix_.setSize(nrows_, ncols_);
}

/* 
 * Go over the Digis and create clusters
 */

void Phase2TrackerClusterizerAlgorithm::clusterizeDetUnit(const edm::DetSet< Phase2TrackerDigi >& digis, Phase2TrackerCluster1DCollectionNew::FastFiller& clusters) {

    // Fill the 2D matrix with the hit information : (hit or not)
    fillMatrix(digis.begin(), digis.end());

    // Number of clusters
    unsigned int numberClusters(0);
    Phase2TrackerDigi firstDigi;
    unsigned int sizeCluster(0);
    bool closeCluster(false); 

    // Loop over the Digis
    // for the S modules, 1 column = 1 strip, so adjacent digis are along the rows
    // same for P modules
    for (unsigned int col(0); col < ncols_; ++col) {
        for (unsigned int row(0); row <  nrows_; ++row) {
            
            // If the Digi is hit 
            if (matrix_(row, col)) {
                // No cluster is open, create a new one
                if (sizeCluster == 0) {
                    // Define first digi 		    
                    firstDigi = Phase2TrackerDigi(row, col);
                    sizeCluster = 1;
                }
                // A cluster is open, increase its size
                else ++sizeCluster;
                // Check if we reached the maximum size of the cluster and need to close it
                closeCluster = ((maxClusterSize_ != 0 and sizeCluster >= maxClusterSize_) ? true : false);
            }
            // Otherwise check if we need to close a cluster (end of cluster)
            else closeCluster = ((sizeCluster != 0) ? true : false);

            // Always close a cluster if we reach the end of the loop
            if (sizeCluster != 0 and row == (nrows_ - 1)) closeCluster = true;

            // If we have to close a cluster, do it
            if (closeCluster) { 
                // Add the cluster to the list
                clusters.push_back(Phase2TrackerCluster1D(firstDigi, sizeCluster));
                // Reset the variables
                sizeCluster = 0;
                // Increase the number of clusters
                ++numberClusters;
            }

            // Check if we hit the maximum number of clusters per module
            if (maxNumberClusters_ != 0 and numberClusters > maxNumberClusters_) return;
        }
    }

    // Reset the matrix
    clearMatrix(digis.begin(), digis.end());
}

/* 
 * Copy the value of the Digis to the 2D matrix (hit or not). 
 */

void Phase2TrackerClusterizerAlgorithm::fillMatrix(edm::DetSet< Phase2TrackerDigi >::const_iterator begin, edm::DetSet< Phase2TrackerDigi >::const_iterator end) {
    for (edm::DetSet< Phase2TrackerDigi >::const_iterator di(begin); di != end; ++di) matrix_.set(di->row(), di->column(), true);
}

/*
 * Clear the array of hits
 */

void Phase2TrackerClusterizerAlgorithm::clearMatrix(edm::DetSet< Phase2TrackerDigi >::const_iterator begin, edm::DetSet< Phase2TrackerDigi >::const_iterator end) {
    for (edm::DetSet< Phase2TrackerDigi >::const_iterator di(begin); di != end; ++di) matrix_.set(di->row(), di->column(), false);
}

