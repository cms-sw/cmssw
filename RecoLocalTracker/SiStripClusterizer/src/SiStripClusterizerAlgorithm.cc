// File: SiStripClusterizerAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgorithm.h"

//using namespace std;

SiStripClusterizerAlgorithm::SiStripClusterizerAlgorithm(const edm::ParameterSet& conf) : 
  conf_(conf),  
  clusterMode_(conf.getParameter<std::string>("ClusterMode")){

  edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizerAlgorithm::SiStripClusterizerAlgorithm] Constructing object...";
  edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizerAlgorithm::SiStripClusterizerAlgorithm] ClusterizingMode: " << clusterMode_;
  
  if ( clusterMode_ == "ThreeThresholdClusterizer" ) {
    ThreeThresholdStripClusterizer_ = new ThreeThresholdStripClusterizer(conf_.getParameter<double>("ChannelThreshold"),
									 conf_.getParameter<double>("SeedThreshold"),
									 conf_.getParameter<double>("ClusterThreshold"),
									 conf_.getParameter<int>("MaxHolesInCluster"));
    validClusterizer_ = true;
  } else {
    edm::LogError("SiStripClusterizer") << "[SiStripClusterizerAlgorithm] No valid strip clusterizer selected, possible clusterizer: ThreeThresholdClusterizer" << std::endl;
    throw edm::Exception(edm::errors::Configuration) << "[SiStripClusterizerAlgorithm] No valid strip clusterizer selected, possible clusterizer: ThreeThresholdClusterizer";
    validClusterizer_ = false;
  }
}

SiStripClusterizerAlgorithm::~SiStripClusterizerAlgorithm() {
  if ( ThreeThresholdStripClusterizer_ != 0 ) {
    delete ThreeThresholdStripClusterizer_;
  }
}

// void SiStripClusterizerAlgorithm::configure( SiStripNoiseService* in) {

//     ThreeThresholdStripClusterizer_->setSiStripNoiseService(in);
// } 


void SiStripClusterizerAlgorithm::run(
const edm::DetSetVector<SiStripDigi>& input,
std::vector<edm::DetSet<SiStripCluster> > & output,
const edm::ESHandle<SiStripNoises> & noiseHandle, 
const edm::ESHandle<SiStripGain> & gainHandle) {

  if ( validClusterizer_ ) {
    int number_detunits          = 0;
    int number_localstriprechits = 0;

    //loop on all detset inside the input collection
    edm::DetSetVector<SiStripDigi>::const_iterator DSViter=input.begin();
    for (; DSViter!=input.end();DSViter++){
      ++number_detunits;
      LogDebug("SiStripClusterizer")  << "[SiStripClusterizerAlgorithm::run] DetID " << DSViter->id;

      edm::DetSet<SiStripCluster> ssc(DSViter->id);
      ThreeThresholdStripClusterizer_->clusterizeDetUnit(*DSViter,ssc, noiseHandle, gainHandle);

      number_localstriprechits += ssc.data.size();
      
      if (ssc.data.size())
	output.push_back(ssc);  // insert the DetSet<SiStripCluster> in the  DetSetVec<SiStripCluster> only if there is at least a digi
    }
    edm::LogInfo("SiStripClusterizer") << "[SiStripClusterizerAlgorithm] execution in mode " << clusterMode_ << " generating " << number_localstriprechits << " SiStripClusters in " << number_detunits << " DetUnits.";
  }
}
