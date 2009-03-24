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
void SiStripDummyAlgo::clusterize( const DigisDSnew& digis, 
				   ClustersV& clusters ) {
  
  // Some initialization
  clusters.clear();
  
  if ( digis.empty() ) { return; }

  DigisV::const_iterator begin( digis.begin() );
  DigisV::const_iterator end( digis.end() );
  SiStripCluster::SiStripDigiRange range( begin, begin );

  // Iterates through digis and builds clusters from non-zero digis
  DigisV::const_iterator idigi( begin );
  DigisV::const_iterator jdigi( end );
  for ( ; idigi != jdigi; ++idigi ) {
    if ( idigi != begin ) {
      if ( idigi->strip() - (idigi-1)->strip() > 1 ) { // check for gap
	clusters.push_back( SiStripCluster( digis.id(), range ) ); 
	range.first = idigi; 
      } 
    }
    range.second = idigi+1;
  }
  if ( range.first != range.second ) { clusters.push_back( SiStripCluster( digis.id(), range ) ); }
    
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

