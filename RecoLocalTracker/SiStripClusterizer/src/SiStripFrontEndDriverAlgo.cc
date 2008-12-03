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
void SiStripFrontEndDriverAlgo::clusterize( const edm::DetSet<SiStripDigi>& digis,
					    edm::DetSetVector<SiStripCluster>& clusters ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripFrontEndDriverAlgo::" << __func__ << "]";
}

// -----------------------------------------------------------------------------
//
void SiStripFrontEndDriverAlgo::add( edm::DetSet<SiStripCluster>& clusters,
				     const uint16_t& strip,
				     const uint16_t& adc ) {
  LogTrace("UNDEFINED_CATEGORY")
    << "[SiStripFrontEndDriverAlgo::" << __func__ << "]";
}

