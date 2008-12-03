#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerFactory.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripDummyAlgo.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripFrontEndDriverAlgo.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripThreeThresholdAlgo.h"
#include "string"

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
  if ( zero_suppr ) { factory_ = NULL; } //@@ to come
  
}

SiStripClusterizerFactory::~SiStripClusterizerFactory() {
  delete algorithm_;
}

void SiStripClusterizerFactory::clusterize( const edm::DetSetVector<SiStripDigi>& digis, edm::DetSetVector<SiStripCluster>& clusters ) {

  edm::DetSetVector<SiStripDigi>::const_iterator idigis = digis.begin();
  for ( ; idigis != digis.end(); idigis++ ) {
    clusterize( *idigis, clusters );
  }
}

void SiStripClusterizerFactory::clusterize( const edm::DetSet<SiStripDigi>& digis, edm::DetSetVector<SiStripCluster>& clusters ) {
  if (algorithm()) algorithm()->clusterize( digis, clusters ); 
}

void SiStripClusterizerFactory::clusterize( const edm::DetSetVector<SiStripRawDigi>& raw_digis, edm::DetSetVector<SiStripCluster>& clusters ) {}

void SiStripClusterizerFactory::clusterize( const edm::DetSet<SiStripRawDigi>& raw_digis, edm::DetSetVector<SiStripCluster>& clusters ) {}

void SiStripClusterizerFactory::eventSetup( const edm::EventSetup& setup) {
  if (algorithm()) algorithm()->eventSetup(setup);
}

