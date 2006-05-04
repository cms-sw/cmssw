// File: SiStripRecHitConverter.cc
// Description:  see SiStripRecHitConverter.h
// Author:  O. Gutsche
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverter.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"



namespace cms
{

  SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf) : 
    recHitConverterAlgorithm_(conf) ,
    conf_(conf)
  {
    produces<SiStripRecHit2DMatchedLocalPosCollection>("matchedRecHit");
    produces<SiStripRecHit2DLocalPosCollection>("rphiRecHit");
    produces<SiStripRecHit2DLocalPosCollection>("stereoRecHit");
  }


  // Virtual destructor needed.
  SiStripRecHitConverter::~SiStripRecHitConverter() { }  

  // Functions that gets called by framework every event
  void SiStripRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es)
  {
    using namespace edm;
    edm::ESHandle<TrackerGeometry> pDD;
    es.get<TrackerDigiGeometryRecord>().get( pDD );
    const TrackerGeometry &tracker(*pDD);
    
    //    edm::ESHandle<MagneticField> pSetup;
    //    es.get<IdealMagneticFieldRecord>().get(pSetup);
    //const MagneticField &BField(*pSetup);

    std::string cpe = conf_.getParameter<std::string>("StripCPE");
    edm::ESHandle<StripClusterParameterEstimator> parameterestimator;
    es.get<TrackerCPERecord>().get(cpe, parameterestimator); 
    const StripClusterParameterEstimator &stripcpe(*parameterestimator);

    std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");

    // Step A: Get Inputs 
    edm::Handle<edm::DetSetVector<SiStripCluster>> clusters;
    e.getByLabel(clusterProducer, clusters);

    // Step B: create empty output collection
    std::auto_ptr<SiStripRecHit2DMatchedLocalPosCollection> outputmatched(new SiStripRecHit2DMatchedLocalPosCollection);
    std::auto_ptr<SiStripRecHit2DLocalPosCollection> outputrphi(new SiStripRecHit2DLocalPosCollection);
    std::auto_ptr<SiStripRecHit2DLocalPosCollection> outputstereo(new SiStripRecHit2DLocalPosCollection);

    // Step C: Invoke the seed finding algorithm
    recHitConverterAlgorithm_.run(*clusters,*outputmatched,*outputrphi,*outputstereo,tracker,stripcpe);

    // Step D: write output to file
    e.put(outputmatched,"matchedRecHit");
    e.put(outputrphi,"rphiRecHit");
    e.put(outputstereo,"stereoRecHit");
  }

}
