// File: SiStripRecHitConverter.cc
// Description:  see SiStripRecHitConverter.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverter.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/TrackingRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


namespace cms
{

  SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf) : 
    recHitConverterAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<SiStripRecHit2DLocalPosCollection>();
  }


  // Virtual destructor needed.
  SiStripRecHitConverter::~SiStripRecHitConverter() { }  

  // Functions that gets called by framework every event
  void SiStripRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es)
  {
    using namespace edm;
    edm::ESHandle<TrackingGeometry> pDD;
    es.get<TrackerDigiGeometryRecord>().get( pDD );
    const TrackingGeometry &tracker(*pDD);

    std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");

    // Step A: Get Inputs 
    edm::Handle<SiStripClusterCollection> clusters;
    e.getByLabel(clusterProducer, clusters);

    // Step B: create empty output collection
    std::auto_ptr<SiStripRecHit2DLocalPosCollection> output(new SiStripRecHit2DLocalPosCollection);

    // Step C: Invoke the seed finding algorithm
    recHitConverterAlgorithm_.run(clusters.product(),*output,tracker);

    // Step D: write output to file
    e.put(output);

  }

}
