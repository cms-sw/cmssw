#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/StripClusterSelectorTopBottom.h"

void StripClusterSelectorTopBottom::produce( edm::Event& event, const edm::EventSetup& setup) {

  edm::Handle< edmNew::DetSetVector<SiStripCluster> > input;
  event.getByLabel(label_, input);

  edm::ESHandle<TrackerGeometry> geom;
  setup.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker( *geom );
  
  std::auto_ptr<edmNew::DetSetVector<SiStripCluster> > output( new edmNew::DetSetVector<SiStripCluster>() );

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator clustSet = input->begin(); clustSet!=input->end(); ++clustSet) {
    edmNew::DetSet<SiStripCluster>::const_iterator clustIt = clustSet->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator end     = clustSet->end();
    
    DetId detIdObject( clustSet->detId() );
    edmNew::DetSetVector<SiStripCluster>::FastFiller spc(*output, detIdObject.rawId());
    const StripGeomDetUnit* theGeomDet = dynamic_cast<const StripGeomDetUnit*> (theTracker.idToDet(detIdObject) );
    const StripTopology * topol = dynamic_cast<const StripTopology*>(&(theGeomDet->specificTopology()));
    
    for(; clustIt!=end;++clustIt) {
      LocalPoint lpclust = topol->localPosition(clustIt->barycenter());
      GlobalPoint GPclust = theGeomDet->surface().toGlobal(Local3DPoint(lpclust.x(),lpclust.y(),lpclust.z()));
      double clustY = GPclust.y();
      if ((clustY * y_) > 0) {
	spc.push_back(*clustIt);
      }
    }
  }
  event.put( output );  
}

DEFINE_FWK_MODULE( StripClusterSelectorTopBottom );
