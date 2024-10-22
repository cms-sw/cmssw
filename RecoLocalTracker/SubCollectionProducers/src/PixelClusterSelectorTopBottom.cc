#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/PixelClusterSelectorTopBottom.h"

void PixelClusterSelectorTopBottom::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<SiPixelClusterCollectionNew> input;
  event.getByToken(token_, input);

  const TrackerGeometry& theTracker = setup.getData(tTrackerGeom_);

  auto output = std::make_unique<SiPixelClusterCollectionNew>();

  for (SiPixelClusterCollectionNew::const_iterator clustSet = input->begin(); clustSet != input->end(); ++clustSet) {
    edmNew::DetSet<SiPixelCluster>::const_iterator clustIt = clustSet->begin();
    edmNew::DetSet<SiPixelCluster>::const_iterator end = clustSet->end();

    DetId detIdObject(clustSet->detId());
    edmNew::DetSetVector<SiPixelCluster>::FastFiller spc(*output, detIdObject);
    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detIdObject));
    const PixelTopology* topol = (&(theGeomDet->specificTopology()));

    for (; clustIt != end; ++clustIt) {
      LocalPoint lpclust = topol->localPosition(MeasurementPoint((*clustIt).x(), (*clustIt).y()));
      GlobalPoint GPclust = theGeomDet->surface().toGlobal(Local3DPoint(lpclust.x(), lpclust.y(), lpclust.z()));
      double clustY = GPclust.y();
      if ((clustY * y_) > 0) {
        spc.push_back(*clustIt);
      }
    }
  }
  event.put(std::move(output));
}

DEFINE_FWK_MODULE(PixelClusterSelectorTopBottom);
