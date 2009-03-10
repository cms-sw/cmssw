#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerFactory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripDummyAlgo.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripFrontEndDriverAlgo.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripThreeThresholdAlgo.h"
#include "string"

// -----------------------------------------------------------------------------
//
SiStripClusterizerFactory::SiStripClusterizerFactory( const edm::ParameterSet& pset ) 
  : algorithms_(),
    algorithm_(0),
    factory_(0)
{

  // Create clusterizer algorithm object
  std::string algo = pset.getUntrackedParameter<std::string>("ClusterizerAlgorithm","DummyAlgorithm");
  if ( algo == "DummyAlgorithm" ) {
    algorithm_ = new SiStripDummyAlgo(pset);
  } else if ( algo == "FrontEndDriver" ) {
    algorithm_ = new SiStripFrontEndDriverAlgo(pset);
  } else if ( algo == "ThreeThreshold" ) {
    algorithm_ = new SiStripThreeThresholdAlgo(pset);
  } else {
    edm::LogWarning("UNDEFINED_CATEGORY")
      << "[SiStripClusterizerFactory::" << __func__ << "]"
      << " Unknown clusterizer specified in .cfg file: \"" 
      << algo 
      << "\". Defaulting to 'FrontEndDriver' algorithm...";
    algorithm_ = new SiStripFrontEndDriverAlgo(pset);
  }

  // Create zero-suppressor factory
  bool zero_suppr = pset.getUntrackedParameter<bool>("PerformZeroSuppression",false);
  if ( zero_suppr ) { 
    edm::LogError("SiStripClusterizerFactory") << " TO BE IMPLEMENTED!";
    factory_ = NULL; //@@ TO BE IMPLEMENTED!!!
  } 
  
}

// -----------------------------------------------------------------------------
//
SiStripClusterizerFactory::~SiStripClusterizerFactory() {
  if ( algorithm_ ) { delete algorithm_; }
}

// -----------------------------------------------------------------------------
//
void SiStripClusterizerFactory::clusterize( const DigisDSV& digis, ClustersDSVnew& clusters ) {
  if ( algorithm_ ) { algorithm_->clusterize( digis, clusters ); }
}

// -----------------------------------------------------------------------------
//
void SiStripClusterizerFactory::clusterize( const DigisDSV& digis, ClustersDSV& clusters ) {
  if ( algorithm_ ) { algorithm_->clusterize( digis, clusters ); }
}

// -----------------------------------------------------------------------------
//
void SiStripClusterizerFactory::clusterize( const RawDigisDSV& digis, ClustersDSVnew& clusters ) {
  //@@ TO BE IMPLEMENTED!!!
  edm::LogError("SiStripClusterizerFactory") << "TO BE IMPLEMENTED!";
}

// -----------------------------------------------------------------------------
//
void SiStripClusterizerFactory::clusterize( const RawDigisDSV& digis, ClustersDSV& clusters ) {
  //@@ TO BE IMPLEMENTED!!!
  edm::LogError("SiStripClusterizerFactory") << "TO BE IMPLEMENTED!";
}

// -----------------------------------------------------------------------------
//
void SiStripClusterizerFactory::eventSetup( const edm::EventSetup& setup) {
  if ( algorithm_ ) { algorithm_->eventSetup(setup); }
}

