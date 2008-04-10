// File: SiStripRecHitConverter.cc
// Description:  see SiStripRecHitConverter.h
// Author:  C.Genta
// Creation Date:  OGU Aug. 1 2005 Initial version.
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverter.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"



namespace cms
{

  SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf) : 
    recHitConverterAlgorithm_(conf) ,
    conf_(conf),
    matchedRecHitsTag_( conf.getParameter<std::string>( "matchedRecHits" ) ), 
    rphiRecHitsTag_( conf.getParameter<std::string>( "rphiRecHits" ) ), 
    stereoRecHitsTag_( conf.getParameter<std::string>( "stereoRecHits" ) )
  {
    produces<SiStripMatchedRecHit2DCollection>( matchedRecHitsTag_ );
    produces<SiStripRecHit2DCollection>( rphiRecHitsTag_ );
    produces<SiStripRecHit2DCollection>( stereoRecHitsTag_ );
  }


  // Virtual destructor needed.
  SiStripRecHitConverter::~SiStripRecHitConverter() { }  

  // Functions that gets called by framework every event
  void SiStripRecHitConverter::produce(edm::Event& e, const edm::EventSetup& es)
  {
    //get tracker geometry
    using namespace edm;
    edm::ESHandle<TrackerGeometry> pDD;
    es.get<TrackerDigiGeometryRecord>().get( pDD );
    const TrackerGeometry &tracker(*pDD);
    
    //get Cluster Parameter Estimator
    std::string cpe = conf_.getParameter<std::string>("StripCPE");
    edm::ESHandle<StripClusterParameterEstimator> parameterestimator;
    es.get<TkStripCPERecord>().get(cpe, parameterestimator); 
    const StripClusterParameterEstimator &stripcpe(*parameterestimator);
    
    //get matcher
    std::string matcher = conf_.getParameter<std::string>("Matcher");
    edm::ESHandle<SiStripRecHitMatcher> rechitmatcher;
    es.get<TkStripCPERecord>().get(matcher, rechitmatcher); 
    const SiStripRecHitMatcher &rhmatcher(*rechitmatcher);
    
    // Step A: Get Inputs 
    std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
    bool regional = conf_.getParameter<bool>("Regional");
    edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
    edm::Handle<edm::SiStripRefGetter<SiStripCluster> > refclusters;
    edm::Handle<edm::SiStripLazyGetter<SiStripCluster> > lazygetter;

    if (regional){
      std::string lazyGetterProducer=conf_.getParameter<std::string>("LazyGetterProducer");
      e.getByLabel(clusterProducer, refclusters);
      e.getByLabel(lazyGetterProducer, lazygetter);
    }
    else e.getByLabel(clusterProducer, clusters);

    // Step B: create empty output collection
    std::auto_ptr<SiStripMatchedRecHit2DCollection> outputmatched(new SiStripMatchedRecHit2DCollection);
    std::auto_ptr<SiStripRecHit2DCollection> outputrphi(new SiStripRecHit2DCollection);
    std::auto_ptr<SiStripRecHit2DCollection> outputstereo(new SiStripRecHit2DCollection);

    // Step C: Invoke the seed finding algorithm
    if (regional) recHitConverterAlgorithm_.run(refclusters,lazygetter,*outputmatched,*outputrphi,*outputstereo,tracker,stripcpe,rhmatcher);
    else recHitConverterAlgorithm_.run(clusters,*outputmatched,*outputrphi,*outputstereo,tracker,stripcpe,rhmatcher);

    // Step D: write output to file
    e.put(outputmatched, matchedRecHitsTag_ );
    e.put(outputrphi, rphiRecHitsTag_ );
    e.put(outputstereo,stereoRecHitsTag_ );
  }

}
