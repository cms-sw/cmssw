#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/TkPixelMeasurementDet.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDetException.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include <iostream>

MeasurementTracker::MeasurementTracker( const edm::EventSetup& setup, 
					const edm::Event& event)
{
  using namespace std;
  initialize( setup);

  cout << "MeasurementTracker: initialize OK" << endl;

  update(event);

  cout << "MeasurementTracker: update OK" << endl;

}

void MeasurementTracker::initialize(const edm::EventSetup& setup)
{
  using namespace std;
    using namespace edm;
    edm::ESHandle<TrackingGeometry> pDD;
    setup.get<TrackerDigiGeometryRecord>().get( pDD );
    const TrackingGeometry &tracker(*pDD);

    cout << "MeasurementTracker::initialize: TrackingGeometry accessed" << endl; 

    const TrackingGeometry::DetContainer& dets = tracker.dets();

    std::cout << "got from TrackingGeometry " << dets.size() << std::endl; 


    for (TrackingGeometry::DetContainer::const_iterator gd=dets.begin();
	 gd != dets.end(); gd++) {

      cout << "GeomDet address " << *gd << endl;
      cout << "GeomDet Surface address " << &(**gd).surface() << endl;

      std::cout << (**gd).surface().position() << std::endl; 
      
      addDet(*gd);
    }
}

void MeasurementTracker::addDet( const GeomDet* gd)
{
  DetId id(gd->geographicalId());
  switch(id.subdetId()){
  case PixelSubdetector::PixelBarrel:
    addPixelDet(gd, pixelCPE);
    break;
  case PixelSubdetector::PixelEndcap:
    addPixelDet(gd, pixelCPE);
    break;
  case StripSubdetector::TIB:
    addStripDet(gd, stripCPE);
    break;
  case StripSubdetector::TID:
    addStripDet(gd, stripCPE);
    break;
  case StripSubdetector::TOB:
    addStripDet(gd, stripCPE);
    break;
  case StripSubdetector::TEC:
    addStripDet(gd, stripCPE);
    break;

    // glued dets should come here too...

  default:
    throw MeasurementDetException("MeasurementTracker ERROR: not a Tracker Subdetector");
  }
}

void MeasurementTracker::addStripDet( const GeomDet* gd,
				      const StripClusterParameterEstimator* cpe)
{
  TkStripMeasurementDet* det = new TkStripMeasurementDet( gd, cpe);
  theStripDets.push_back(det);
  theDetMap[gd->geographicalId()] = det;
}

void MeasurementTracker::addPixelDet( const GeomDet* gd,
				      const PixelClusterParameterEstimator* cpe)
{
  TkPixelMeasurementDet* det = new TkPixelMeasurementDet( gd, cpe);
  thePixelDets.push_back(det);
  theDetMap[gd->geographicalId()] = det;
}

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

void MeasurementTracker::update( const edm::Event& event) const
{
  typedef SiStripClusterCollection::Range    StripClusterRange;
  typedef SiPixelClusterCollection::Range    PixelClusterRange;

  // std::string clusterProducer = conf_.getParameter<std::string>("ClusterProducer");
  std::string stripClusterProducer ("ClusterProducer"); // FIXME
  edm::Handle<SiStripClusterCollection> clusterHandle;
  event.getByLabel(stripClusterProducer, clusterHandle);
  const SiStripClusterCollection* clusterCollection = clusterHandle.product();

  // loop over all strip dets
  for (std::vector<TkStripMeasurementDet*>::const_iterator i=theStripDets.begin();
       i!=theStripDets.end(); i++) {
    // foreach det get cluster range
    StripClusterRange range = clusterCollection->get( (**i).geomDet().geographicalId().rawId());
    // push cluster range in det
    (**i).update( range);
  }


  // Pixel Clusters
  std::string pixelClusterProducer ("ClusterProducer"); // FIXME
  edm::Handle<SiPixelClusterCollection> pixelClusters;
  event.getByLabel(pixelClusterProducer, pixelClusters);
  const SiPixelClusterCollection * pixelCollection = pixelClusters.product();
  for (std::vector<TkPixelMeasurementDet*>::const_iterator i=thePixelDets.begin();
       i!=thePixelDets.end(); i++) {
    // foreach det get cluster range
    PixelClusterRange range = pixelCollection->get( (**i).geomDet().geographicalId().rawId());
    // push cluster range in det
    (**i).update( range);
  }


  /// or maybe faster: loop over all strip dets and clear them
  /// loop over dets with clusters and set range

}

const MeasurementDet* MeasurementTracker::measurementDet(const DetId& id) const
{
  return 0;
}
