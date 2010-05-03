#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripFrontEndDriverAlgo.h"

// -----------------------------------------------------------------------------
//
SiStripFrontEndDriverAlgo::SiStripFrontEndDriverAlgo( const edm::ParameterSet& pset ) 
  : SiStripClusterizerAlgo(pset) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripFrontEndDriverAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
SiStripFrontEndDriverAlgo::~SiStripFrontEndDriverAlgo() {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripFrontEndDriverAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripFrontEndDriverAlgo::clusterize( const DigisDSnew& digis, 
					    ClustersV& clusters ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripFrontEndDriverAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripFrontEndDriverAlgo::add( ClustersV& data, 
				     const uint32_t& id, 
				     const uint16_t& istrip, 
				     const uint16_t& adc ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripFrontEndDriverAlgo::" << __func__ << "]";
}

