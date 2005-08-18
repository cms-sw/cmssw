// File: SiStripClusterizer.cc
// Description:  see SiStripClusterizer.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterizer.h"
#include "DataFormats/SiStripDigi/interface/StripDigiCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms
{

  SiStripClusterizer::SiStripClusterizer(edm::ParameterSet const& conf) : 
    siStripClusterizerAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<SiStripClusterCollection>();
  }


  // Virtual destructor needed.
  SiStripClusterizer::~SiStripClusterizer() { }  

  // Functions that gets called by framework every event
  void SiStripClusterizer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // retrieve producer name of input StripDigiCollection
    std::string digiProducer = conf_.getParameter<std::string>("DigiProducer");

    // Step A: Get Inputs 
    edm::Handle<StripDigiCollection> stripDigis;
    e.getByLabel(digiProducer, stripDigis);

    // Step B: create empty output collection
    std::auto_ptr<SiStripClusterCollection> output(new SiStripClusterCollection);

    // Step C: Invoke the strip clusterizer algorithm
    siStripClusterizerAlgorithm_.run(stripDigis.product(),*output);

    // Step D: write output to file
    e.put(output);

  }

}
