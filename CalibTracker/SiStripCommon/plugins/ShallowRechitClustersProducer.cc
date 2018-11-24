#include "CalibTracker/SiStripCommon/interface/ShallowRechitClustersProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

ShallowRechitClustersProducer::ShallowRechitClustersProducer(const edm::ParameterSet& iConfig)
  :  Suffix       ( iConfig.getParameter<std::string>("Suffix") ),
     Prefix       ( iConfig.getParameter<std::string>("Prefix") ),
     clusters_token_( consumes< edmNew::DetSetVector<SiStripCluster> >(iConfig.getParameter<edm::InputTag>("Clusters")))
{
	std::vector<edm::InputTag> rec_hits_tags = iConfig.getParameter<std::vector<edm::InputTag> >("InputTags");
	for(auto itag : rec_hits_tags) {
		rec_hits_tokens_.push_back(
			consumes< SiStripRecHit2DCollection >(itag)
			);
	}

  produces <std::vector<float> >        ( Prefix + "strip"      + Suffix );   
  produces <std::vector<float> >        ( Prefix + "merr"       + Suffix );   
  produces <std::vector<float> >        ( Prefix + "localx"     + Suffix );   
  produces <std::vector<float> >        ( Prefix + "localy"     + Suffix );   
  produces <std::vector<float> >        ( Prefix + "localxerr"  + Suffix );   
  produces <std::vector<float> >        ( Prefix + "localyerr"  + Suffix );   
  produces <std::vector<float> >        ( Prefix + "globalx"    + Suffix );   
  produces <std::vector<float> >        ( Prefix + "globaly"    + Suffix );   
  produces <std::vector<float> >        ( Prefix + "globalz"    + Suffix );   
}

void ShallowRechitClustersProducer::
produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  shallow::CLUSTERMAP clustermap = shallow::make_cluster_map(iEvent, clusters_token_); 

  int size = clustermap.size();
  auto  strip       = std::make_unique <std::vector<float>>(size,  -10000  );   
  auto  merr        = std::make_unique <std::vector<float>>(size,  -10000  );   
  auto  localx      = std::make_unique <std::vector<float>>(size,  -10000  );   
  auto  localy      = std::make_unique <std::vector<float>>(size,  -10000  );   
  auto  localxerr   = std::make_unique <std::vector<float>>(size,  -1  );   
  auto  localyerr   = std::make_unique <std::vector<float>>(size,  -1  );     
  auto  globalx     = std::make_unique <std::vector<float>>(size,  -10000  );   
  auto  globaly     = std::make_unique <std::vector<float>>(size,  -10000  );   
  auto  globalz     = std::make_unique <std::vector<float>>(size,  -10000  );   

  edm::ESHandle<TrackerGeometry> theTrackerGeometry; iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );

	for(auto recHit_token : rec_hits_tokens_) {
		edm::Handle<SiStripRecHit2DCollection> recHits; 
		iEvent.getByToken(recHit_token, recHits);
    for(auto const& ds : *recHits) {
      for(auto const& hit : ds) {
	
				shallow::CLUSTERMAP::iterator cluster = clustermap.find( std::make_pair(hit.geographicalId().rawId(), hit.cluster()->firstStrip()   ) );
				if(cluster != clustermap.end() ) {
					const StripGeomDetUnit* theStripDet = dynamic_cast<const StripGeomDetUnit*>( theTrackerGeometry->idToDet( hit.geographicalId() ) );
					unsigned int i = cluster->second;
					strip->at(i)  =    theStripDet->specificTopology().strip(hit.localPosition());
					merr->at(i)   =    sqrt(theStripDet->specificTopology().measurementError(hit.localPosition(), hit.localPositionError()).uu());
					localx->at(i) =    hit.localPosition().x();
					localy->at(i) =    hit.localPosition().y();
					localxerr->at(i) = sqrt(hit.localPositionError().xx());
					localyerr->at(i) = sqrt(hit.localPositionError().yy());
					globalx->at(i) =   theStripDet->toGlobal(hit.localPosition()).x();
					globaly->at(i) =   theStripDet->toGlobal(hit.localPosition()).y();
					globalz->at(i) =   theStripDet->toGlobal(hit.localPosition()).z();
				}
				else {throw cms::Exception("cluster not found");}
      }
    }
  }
  
  iEvent.put(std::move(strip),      Prefix + "strip"      + Suffix );   
  iEvent.put(std::move(merr),       Prefix + "merr"       + Suffix );   
  iEvent.put(std::move(localx),     Prefix + "localx"     + Suffix );   
  iEvent.put(std::move(localy),     Prefix + "localy"     + Suffix );   
  iEvent.put(std::move(localxerr),  Prefix + "localxerr"  + Suffix );   
  iEvent.put(std::move(localyerr),  Prefix + "localyerr"  + Suffix );   
  iEvent.put(std::move(globalx),    Prefix + "globalx"    + Suffix );   
  iEvent.put(std::move(globaly),    Prefix + "globaly"    + Suffix );   
  iEvent.put(std::move(globalz),    Prefix + "globalz"    + Suffix );   
}
