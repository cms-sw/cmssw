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
  
//   SiStripCluster::SiStripDigiRange range( digis.end(), digis.end() );
  
//   // Iterates through digis and builds clusters from non-zero digis
//   DigisV::const_iterator idigi = digis_begin;
//   DigisV::const_iterator jdigi = digis_end;
//   for ( ; idigi != jdigi; ++idigi ) {
//     if ( idigi == digis_begin ) { // first digi
//       range.first = idigi; 
//       range.second = idigi+1; 
//     } else if ( idigi+1 == digis_end ) { // last digi
//       range.second++;
//       clusters.push_back( SiStripCluster( id, range ) ); 
//     } else { // all other digis
//       if ( idigi->strip() - (idigi-1)->strip() == 1 ) { range.second++; } 
//       else { 
// 	clusters.push_back( SiStripCluster( id, range ) ); 
// 	range.first = idigi; 
// 	range.second = idigi+1; 
//       }
//     }
//   }
  
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

