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


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitConverter.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"

#include "DataFormats/Common/interface/DetSet2RangeMap.h"



namespace cms
{

  SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf) : 
    recHitConverterAlgorithm_(conf) ,
    conf_(conf),
    matchedRecHitsTag_( conf.getParameter<std::string>( "matchedRecHits" ) ), 
    rphiRecHitsTag_( conf.getParameter<std::string>( "rphiRecHits" ) ), 
    stereoRecHitsTag_( conf.getParameter<std::string>( "stereoRecHits" ) ),
    np_("New"),
    m_newCont(conf.getUntrackedParameter<bool>("newContainer",false))
  {
    produces<SiStripMatchedRecHit2DCollection>( matchedRecHitsTag_ );
    produces<SiStripRecHit2DCollection>( rphiRecHitsTag_ );
    produces<SiStripRecHit2DCollection>( stereoRecHitsTag_ );

    if (m_newCont) {
      produces<SiStripMatchedRecHit2DCollectionNew>( matchedRecHitsTag_+np_ );
      produces<SiStripRecHit2DCollectionNew>( rphiRecHitsTag_+np_ );
      produces<SiStripRecHit2DCollectionNew>( stereoRecHitsTag_+np_ );
    }
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
    edm::Handle<edm::DetSetVector<SiStripCluster> > clusters;
    edm::Handle<edm::SiStripRefGetter<SiStripCluster> > refclusters;

    if (regional) e.getByLabel(clusterProducer, refclusters);
    else e.getByLabel(clusterProducer, clusters);

    // Step B: create empty output collection
    std::auto_ptr<SiStripMatchedRecHit2DCollectionNew> outputmatched(new SiStripMatchedRecHit2DCollectionNew);
    std::auto_ptr<SiStripRecHit2DCollectionNew> outputrphi(new SiStripRecHit2DCollectionNew);
    std::auto_ptr<SiStripRecHit2DCollectionNew> outputstereo(new SiStripRecHit2DCollectionNew);

    // Step C: Invoke the seed finding algorithm
    if (regional) recHitConverterAlgorithm_.run(refclusters,*outputmatched,*outputrphi,*outputstereo,tracker,stripcpe,rhmatcher);
    else recHitConverterAlgorithm_.run(clusters,*outputmatched,*outputrphi,*outputstereo,tracker,stripcpe,rhmatcher);

   // Step Z: temporary write also the old collection
    std::auto_ptr<SiStripMatchedRecHit2DCollection> outputmatchedOld(new SiStripMatchedRecHit2DCollection);
    std::auto_ptr<SiStripRecHit2DCollection> outputrphiOld(new SiStripRecHit2DCollection);
    std::auto_ptr<SiStripRecHit2DCollection> outputstereoOld(new SiStripRecHit2DCollection);

    edmNew::copy(*outputmatched,*outputmatchedOld);
    edmNew::copy(*outputrphi,*outputrphiOld);
    edmNew::copy(*outputstereo,*outputstereoOld);
    e.put(outputmatchedOld, matchedRecHitsTag_);
    e.put(outputrphiOld, rphiRecHitsTag_);
    e.put(outputstereoOld,stereoRecHitsTag_);
 
    // Step D: write output to file
    if (m_newCont) {
      e.put(outputmatched, matchedRecHitsTag_+np_ );
      e.put(outputrphi, rphiRecHitsTag_+np_ );
      e.put(outputstereo,stereoRecHitsTag_+np_ );
    }
  }

}
