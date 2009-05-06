#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitConverter.h"

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf) 
  : recHitConverterAlgorithm_(conf) ,
    conf_(conf),
    matchedRecHitsTag_( conf.getParameter<std::string>( "matchedRecHits" ) ), 
    rphiRecHitsTag_( conf.getParameter<std::string>( "rphiRecHits" ) ), 
    stereoRecHitsTag_( conf.getParameter<std::string>( "stereoRecHits" ) )
{
  produces<SiStripMatchedRecHit2DCollection>( matchedRecHitsTag_ );
  produces<SiStripRecHit2DCollection>( rphiRecHitsTag_ );
  produces<SiStripRecHit2DCollection>( stereoRecHitsTag_ );
  produces<SiStripRecHit2DCollection>( rphiRecHitsTag_ + "Unmatched" );
  produces<SiStripRecHit2DCollection>( stereoRecHitsTag_ +  "Unmatched" );
}


void SiStripRecHitConverter::
produce(edm::Event& e, const edm::EventSetup& es)
{
  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord>().get( pDD );
  const TrackerGeometry &tracker(*pDD);
  
  edm::ESInputTag cpe = conf_.getParameter<edm::ESInputTag>("StripCPE");
  edm::ESHandle<StripClusterParameterEstimator> parameterestimator;
  es.get<TkStripCPERecord>().get(cpe, parameterestimator); 
  const StripClusterParameterEstimator &stripcpe(*parameterestimator);
  
  edm::ESInputTag matcher= conf_.getParameter<edm::ESInputTag>("Matcher");
  edm::ESHandle<SiStripRecHitMatcher> rechitmatcher;
  es.get<TkStripCPERecord>().get(matcher, rechitmatcher); 
  const SiStripRecHitMatcher &rhmatcher(*rechitmatcher);
  
  //maybe get the SiStripQuality
  const SiStripQuality *ptr_stripQuality = 0;
  edm::ESHandle<SiStripQuality> stripQuality;
  if (conf_.existsAs<bool>("useSiStripQuality") && conf_.getParameter<bool>("useSiStripQuality")) {
    edm::ESInputTag qualityTag = conf_.getParameter<edm::ESInputTag>("siStripQualityLabel");
    es.get<SiStripQualityRcd>().get(qualityTag, stripQuality);
    ptr_stripQuality = stripQuality.product();
  }
  
  // Step A: Get Inputs 
  edm::InputTag clusterProducer = conf_.getParameter<edm::InputTag>("ClusterProducer");
  bool regional = conf_.getParameter<bool>("Regional");
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
  edm::Handle<edm::RefGetter<SiStripCluster> > refclusters;
  edm::Handle<edm::LazyGetter<SiStripCluster> > lazygetter;
  
  if (regional){
    edm::InputTag lazyGetterProducer=conf_.getParameter<edm::InputTag>("LazyGetterProducer");
    e.getByLabel(clusterProducer, refclusters);
    e.getByLabel(lazyGetterProducer, lazygetter);
  }
  else e.getByLabel(clusterProducer, clusters);
  
  // Step B: create empty output collection
  std::auto_ptr<SiStripMatchedRecHit2DCollection> outputmatched(new SiStripMatchedRecHit2DCollection);
  std::auto_ptr<SiStripRecHit2DCollection> outputrphi(new SiStripRecHit2DCollection);
  std::auto_ptr<SiStripRecHit2DCollection> outputstereo(new SiStripRecHit2DCollection);
  std::auto_ptr<SiStripRecHit2DCollection> outputrphiUnmatched(new SiStripRecHit2DCollection);
  std::auto_ptr<SiStripRecHit2DCollection> outputstereoUnmatched(new SiStripRecHit2DCollection);
  
  // Step C: Invoke the seed finding algorithm
  if (regional) {
    recHitConverterAlgorithm_.run(refclusters,lazygetter,
				  *outputmatched,*outputrphi,*outputstereo,*outputrphiUnmatched,*outputstereoUnmatched,
				  tracker,stripcpe,rhmatcher,ptr_stripQuality);
  } else {
    recHitConverterAlgorithm_.run(clusters,
				  *outputmatched,*outputrphi,*outputstereo,*outputrphiUnmatched,*outputstereoUnmatched,
				  tracker,stripcpe,rhmatcher,ptr_stripQuality);
  }
  
  // Step D: write output to file
  e.put(outputmatched, matchedRecHitsTag_ );
  e.put(outputrphi, rphiRecHitsTag_ );
  e.put(outputstereo,stereoRecHitsTag_ );
  e.put(outputrphiUnmatched, rphiRecHitsTag_ + "Unmatched" );
  e.put(outputstereoUnmatched, stereoRecHitsTag_ + "Unmatched" );
  
}
