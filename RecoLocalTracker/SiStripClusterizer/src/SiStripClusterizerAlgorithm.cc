// File: SiStripClusterizerAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <iostream>

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizerAlgorithm.h"

#include "DataFormats/SiStripDigi/interface/StripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdStripClusterizer.h"

using namespace std;

SiStripClusterizerAlgorithm::SiStripClusterizerAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
  
  clusterMode_ = conf_.getParameter<string>("ClusterMode");

  if ( clusterMode_ == "ThreeThresholdClusterizer" ) {
    threeThreshold_ = new ThreeThresholdStripClusterizer(conf_.getParameter<double>("ChannelThreshold"),
							 conf_.getParameter<double>("SeedThreshold"),
							 conf_.getParameter<double>("ClusterThreshold"));
    validClusterizer_ = true;
  } else {
    std::cout << "[SiStripClusterizerAlgorithm] No valid strip clusterizer selected, possible clusterizer: ThreeThresholdClusterizer" << endl;
    validClusterizer_ = false;
  }
}

SiStripClusterizerAlgorithm::~SiStripClusterizerAlgorithm() {

  if ( threeThreshold_ != 0 ) {
    delete threeThreshold_;
  }

}


void SiStripClusterizerAlgorithm::run(const StripDigiCollection* input, SiStripClusterCollection &output)
{

  if ( validClusterizer_ ) {
    int number_detunits          = 0;
    int number_localstriprechits = 0;

    // get vector of detunit ids
    const std::vector<unsigned int> detIDs = input->detIDs();
    
    // loop over detunits
    for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); ++detunit_iterator ) {
      unsigned int id = *detunit_iterator;
      ++number_detunits;
      const StripDigiCollection::Range digiRange = input->get(id);
      StripDigiCollection::ContainerIterator digiRangeIteratorBegin = digiRange.first;
      StripDigiCollection::ContainerIterator digiRangeIteratorEnd   = digiRange.second;
      
      if ( clusterMode_ == "ThreeThresholdClusterizer" ) {
	
	// dummies, right now, all empty
	vector<float> noiseVec(768,2);
	vector<short> badChannels;
	
	vector<SiStripCluster> collector = threeThreshold_->clusterizeDetUnit(digiRangeIteratorBegin, 
										digiRangeIteratorEnd,
										id,
										noiseVec,
										badChannels);

	SiStripClusterCollection::Range inputRange;
	inputRange.first = collector.begin();
	inputRange.second = collector.end();
	output.put(inputRange,id);
	number_localstriprechits += collector.size();
      }
    }

    if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
      cout << "[SiStripClusterizerAlgorithm] execution in mode " << clusterMode_ << " generating " << number_localstriprechits << " SiStripClusters in " << number_detunits << " DetUnits." << endl; 
    }
  }
  

};
