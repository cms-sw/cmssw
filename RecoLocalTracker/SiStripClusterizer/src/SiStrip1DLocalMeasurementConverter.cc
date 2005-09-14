// File: SiStrip1DLocalMeasurementConverter.cc
// Description:  see SiStrip1DLocalMeasurementConverter.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStrip1DLocalMeasurementConverter.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStrip1DLocalMeasurementCollection.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{

  SiStrip1DLocalMeasurementConverter::SiStrip1DLocalMeasurementConverter(edm::ParameterSet const& conf) : 
    siStrip1DLocalMeasurementConverterAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<SiStrip1DLocalMeasurementCollection>();
  }


  // Virtual destructor needed.
  SiStrip1DLocalMeasurementConverter::~SiStrip1DLocalMeasurementConverter() { }  

  // Functions that gets called by framework every event
  void SiStrip1DLocalMeasurementConverter::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input StripDigiCollection
    std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");

    // Step A: Get Inputs 
    edm::Handle<SiStripClusterCollection> stripClusters;
    e.getByLabel(clusterProducer, stripClusters);

    // Step B: create empty output collection
    std::auto_ptr<SiStrip1DLocalMeasurementCollection> output(new SiStrip1DLocalMeasurementCollection);

    // Step C: Invoke the algorithm
    siStrip1DLocalMeasurementConverterAlgorithm_.run(stripClusters.product(),es, *output);

    // Step D: write output to file
    e.put(output);

  }

}
