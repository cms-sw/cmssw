#include "CalibTracker/SiStripCommon/interface/ShallowRechitClustersProducer.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"
#include "boost/foreach.hpp"

ShallowRechitClustersProducer::ShallowRechitClustersProducer(const edm::ParameterSet& iConfig)
  :  Suffix       ( iConfig.getParameter<std::string>("Suffix") ),
     Prefix       ( iConfig.getParameter<std::string>("Prefix") ),
     theClustersLabel( iConfig.getParameter<edm::InputTag>("Clusters")),
     inputTags    ( iConfig.getParameter<std::vector<edm::InputTag> >("InputTags"))
{
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
  shallow::CLUSTERMAP clustermap = shallow::make_cluster_map(iEvent,theClustersLabel);

  int size = clustermap.size();
  std::auto_ptr<std::vector<float> >  strip       ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  merr        ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  localx      ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  localy      ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  localxerr   ( new std::vector<float>(size,  -1  ));   
  std::auto_ptr<std::vector<float> >  localyerr   ( new std::vector<float>(size,  -1  ));     
  std::auto_ptr<std::vector<float> >  globalx     ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  globaly     ( new std::vector<float>(size,  -10000  ));   
  std::auto_ptr<std::vector<float> >  globalz     ( new std::vector<float>(size,  -10000  ));   

  edm::ESHandle<TrackerGeometry> theTrackerGeometry; iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );

  BOOST_FOREACH(const edm::InputTag& input, inputTags ) {  edm::Handle<SiStripRecHit2DCollection> recHits; iEvent.getByLabel(input, recHits);
    BOOST_FOREACH( const SiStripRecHit2DCollection::value_type& ds, *recHits) {
      BOOST_FOREACH( const SiStripRecHit2D& hit, ds) {
	
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
  
  iEvent.put( strip,       Prefix + "strip"      + Suffix );   
  iEvent.put( merr,        Prefix + "merr"       + Suffix );   
  iEvent.put( localx ,     Prefix + "localx"     + Suffix );   
  iEvent.put( localy ,     Prefix + "localy"     + Suffix );   
  iEvent.put( localxerr ,  Prefix + "localxerr"  + Suffix );   
  iEvent.put( localyerr ,  Prefix + "localyerr"  + Suffix );   
  iEvent.put( globalx ,    Prefix + "globalx"    + Suffix );   
  iEvent.put( globaly ,    Prefix + "globaly"    + Suffix );   
  iEvent.put( globalz ,    Prefix + "globalz"    + Suffix );   
}
