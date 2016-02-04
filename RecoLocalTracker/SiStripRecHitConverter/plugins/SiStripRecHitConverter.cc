#include "RecoLocalTracker/SiStripRecHitConverter/plugins/SiStripRecHitConverter.h"
#include "FWCore/Framework/interface/Event.h"

SiStripRecHitConverter::SiStripRecHitConverter(edm::ParameterSet const& conf) 
  : recHitConverterAlgorithm(conf) ,
    matchedRecHitsTag( conf.getParameter<std::string>( "matchedRecHits" ) ), 
    rphiRecHitsTag( conf.getParameter<std::string>( "rphiRecHits" ) ), 
    stereoRecHitsTag( conf.getParameter<std::string>( "stereoRecHits" ) ),
    clusterProducer(conf.getParameter<edm::InputTag>("ClusterProducer")),
    lazyGetterProducer(conf.getParameter<edm::InputTag>("LazyGetterProducer")),
    regional(conf.getParameter<bool>("Regional"))
{
  produces<SiStripMatchedRecHit2DCollection>( matchedRecHitsTag );
  produces<SiStripRecHit2DCollection>( rphiRecHitsTag );
  produces<SiStripRecHit2DCollection>( stereoRecHitsTag );
  produces<SiStripRecHit2DCollection>( rphiRecHitsTag + "Unmatched" );
  produces<SiStripRecHit2DCollection>( stereoRecHitsTag +  "Unmatched" );
}

void SiStripRecHitConverter::
produce(edm::Event& e, const edm::EventSetup& es)
{
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > clusters;
  edm::Handle<edm::RefGetter<SiStripCluster> > refclusters;
  edm::Handle<edm::LazyGetter<SiStripCluster> > lazygetter;
  
  if (regional){
    e.getByLabel(clusterProducer, refclusters);
    e.getByLabel(lazyGetterProducer, lazygetter);
  } else e.getByLabel(clusterProducer, clusters);
  
  SiStripRecHitConverterAlgorithm::products output;
  recHitConverterAlgorithm.initialize(es);

  if (regional) recHitConverterAlgorithm.run(refclusters,lazygetter,output);
  else          recHitConverterAlgorithm.run(clusters, output);
  
  LogDebug("SiStripRecHitConverter") << "found\n"  
				     << output.rphi->dataSize()   << "  clusters in mono detectors\n"                            
				     << output.stereo->dataSize() << "  clusters in partners stereo detectors\n";

  e.put( output.matched,         matchedRecHitsTag );
  e.put( output.rphi,            rphiRecHitsTag    );
  e.put( output.stereo,          stereoRecHitsTag  );
  e.put( output.rphiUnmatched,   rphiRecHitsTag   + "Unmatched" );
  e.put( output.stereoUnmatched, stereoRecHitsTag + "Unmatched" );  

}
