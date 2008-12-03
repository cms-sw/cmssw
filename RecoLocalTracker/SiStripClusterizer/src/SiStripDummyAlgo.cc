#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripDummyAlgo.h"
#include <iostream>
#include <sstream>
#include <vector>

// -----------------------------------------------------------------------------
//
SiStripDummyAlgo::SiStripDummyAlgo( const edm::ParameterSet& pset ) 
  : SiStripClusterizerAlgo(pset) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
SiStripDummyAlgo::~SiStripDummyAlgo() {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripDummyAlgo::clusterize( const edm::DetSet<SiStripDigi>& digis,
				   edm::DetSetVector<SiStripCluster>& clusters ) {

  // Some initialization
  edm::DetSet<SiStripCluster> clus( digis.id );
  SiStripCluster::SiStripDigiRange end( digis.data.end(), digis.data.end() );
  SiStripCluster::SiStripDigiRange range = end;

  // Iterates through digis and builds clusters from non-zero digis
  edm::DetSet<SiStripDigi>::const_iterator idigi = digis.data.begin();
  for ( ; idigi != digis.data.end(); idigi++ ) {
    if ( idigi == digis.data.begin() ) { // first digi
      range.first = idigi; 
      range.second = idigi+1; 
    } else if ( idigi+1 == digis.data.end() ) { // last digi
      range.second++;
      clus.push_back( SiStripCluster(digis.id,range) ); 
    } else { // all other digis
      if ( idigi->strip() - (idigi-1)->strip() == 1 ) { range.second++; } 
      else { 
	clus.push_back( SiStripCluster(digis.id,range) ); 
	range.first = idigi; 
	range.second = idigi+1; 
      }
    }
  }

  // Copy DetSet into DetSetVector if clusters found
  if ( !clus.data.empty() ) { clusters.insert(clus); }
  
}

// -----------------------------------------------------------------------------
//
void SiStripDummyAlgo::add( edm::DetSet<SiStripCluster>& clusters,
			    const uint16_t& strip,
			    const uint16_t& adc ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}

