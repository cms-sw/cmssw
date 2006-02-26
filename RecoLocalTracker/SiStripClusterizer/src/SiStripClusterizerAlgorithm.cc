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

#include "CondFormats/SiStripObjects/interface/SiStripNoise.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetType.h"
#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"


using namespace std;

SiStripClusterizerAlgorithm::SiStripClusterizerAlgorithm(const edm::ParameterSet& conf) : 
  conf_(conf),  
  clusterMode_(conf.getParameter<string>("ClusterMode")),
  ElectronsPerADC_(conf.getParameter<double>("ElectronPerAdc")),
  ENC_(conf.getParameter<double>("EquivalentNoiseCharge300um")),
  UseNoiseBadStripFlagFromDB_(conf.getParameter<bool>("UseNoiseBadStripFlagFromDB")){
  
  if ( clusterMode_ == "ThreeThresholdClusterizer" ) {
    threeThreshold_ = new ThreeThresholdStripClusterizer(conf_.getParameter<double>("ChannelThreshold"),
							 conf_.getParameter<double>("SeedThreshold"),
							 conf_.getParameter<double>("ClusterThreshold")
							 conf_.getParameter<int>("MaxHolesInCluster"));
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

void SiStripClusterizerAlgorithm::run(const StripDigiCollection* input, SiStripClusterCollection &output,
				      const edm::ESHandle<SiStripNoises>& noise, const edm::ESHandle<TrackingGeometry>& pDD)
{
  if ( validClusterizer_ ) {
    int number_detunits          = 0;
    int number_localstriprechits = 0;

    // get vector of detunit ids
    const std::vector<unsigned int> detIDs = input->detIDs();
    
    // loop over detunits
    for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); ++detunit_iterator ) {
      unsigned int detID = *detunit_iterator;
      ++number_detunits;
      const StripDigiCollection::Range digiRange = input->get(detID);
      StripDigiCollection::ContainerIterator digiRangeIteratorBegin = digiRange.first;
      StripDigiCollection::ContainerIterator digiRangeIteratorEnd   = digiRange.second;
      
      if ( clusterMode_ == "ThreeThresholdClusterizer" ) {

	if (UseNoiseBadStripFlagFromDB_==false){

	  //Case of SingleValueNoiseValue for all strips of a Detector
	  //Noise is proportional to sensor depth

	  std::cout << "Using a SingleNoiseValue and good strip flags" << std::endl;

	  //FIXME
	  // Access info for DetID from the geometry
	  
	  pDD->idToDet(detID);
	  for(TrackingGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){

	    if( dynamic_cast<StripGeomDetUnit*>((*it))!=0){

	      uint32_t detid=((*it)->geographicalId()).rawId();

	      const StripTopology& st = dynamic_cast<StripGeomDetUnit*>((*it))->specificTopology();
	      int numStrips = st.nstrips();  // det module number of strips
	      //thickness:
	      const BoundSurface& bs = (dynamic_cast<StripGeomDetUnit*>((det)))->surface();
	      moduleThickness = bs.bounds().thickness();	
	 
	  //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

	  float noise = ENC*moduleThickness/(0.03)/ElectronsPerADC_;
	  vector<float> noiseVec(numStrips,noise);
	  
	  //FIXME 
	  // for fixed noise value all strips are goods!!!
	  vector<short> badChannels(numStrips,0);
	  /////////////////////////////////////
	
	  //Construct a SiStripNoiseVector in order to be compliant with the DB access


	} else {

	  //Case of Noise and BadStrip flags access from DB
	  std::cout << "Using Noise and BadStrip flags accessed from DB" << std::endl;

	  const SiStripNoiseVector& vnoise = noise->getSiStripNoiseVector(detID);
	  
	  if (vnoise.size() <= 0)
	    {
	      std::cout << "WARNING requested Noise Vector for detID " << detID << " that isn't in map " << std::endl; 
	      continue;
	    }
	}
	
	vector<SiStripCluster> collector = threeThreshold_->clusterizeDetUnit(digiRangeIteratorBegin, 
									      digiRangeIteratorEnd,
									      detID,
									      vnoise);

	SiStripClusterCollection::Range inputRange;
	inputRange.first = collector.begin();
	inputRange.second = collector.end();
	output.put(inputRange,detID);
	number_localstriprechits += collector.size();
      }
    }
    cout << "[SiStripClusterizerAlgorithm] execution in mode " << clusterMode_ << " generating " << number_localstriprechits << " SiStripClusters in " << number_detunits << " DetUnits." << endl; 
  }
};
