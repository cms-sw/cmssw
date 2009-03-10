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
void SiStripDummyAlgo::clusterize( const DigisDS& digis,
				   ClustersDS& clusters ) {
  
  // Some initialization
  clusters.data.clear();
  
  SiStripCluster::SiStripDigiRange end( digis.data.end(), digis.data.end() );
  SiStripCluster::SiStripDigiRange range = end;
  
  // Iterates through digis and builds clusters from non-zero digis
  edm::DetSet<SiStripDigi>::const_iterator idigi = digis.data.begin();
  edm::DetSet<SiStripDigi>::const_iterator jdigi = digis.data.end();
  for ( ; idigi != jdigi; ++idigi ) {
    if ( idigi == digis.data.begin() ) { // first digi
      range.first = idigi; 
      range.second = idigi+1; 
    } else if ( idigi+1 == digis.data.end() ) { // last digi
      range.second++;
      clusters.push_back( SiStripCluster(digis.id,range) ); 
    } else { // all other digis
      if ( idigi->strip() - (idigi-1)->strip() == 1 ) { range.second++; } 
      else { 
	clusters.push_back( SiStripCluster(digis.id,range) ); 
	range.first = idigi; 
	range.second = idigi+1; 
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripDummyAlgo::add( ClustersV& clusters,
			    const uint32_t& id, 
			    const uint16_t& strip,
			    const uint16_t& adc ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripDummyAlgo::" << __func__ << "]";
}

