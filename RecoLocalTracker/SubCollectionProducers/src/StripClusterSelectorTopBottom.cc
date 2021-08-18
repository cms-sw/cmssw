#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalTracker/SubCollectionProducers/interface/StripClusterSelectorTopBottom.h"

void StripClusterSelectorTopBottom::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<edmNew::DetSetVector<SiStripCluster>> input;
  event.getByToken(token_, input);

  const TrackerGeometry& theTracker = setup.getData(tTrackerGeom_);

  auto output = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();

  for (edmNew::DetSetVector<SiStripCluster>::const_iterator clustSet = input->begin(); clustSet != input->end();
       ++clustSet) {
    edmNew::DetSet<SiStripCluster>::const_iterator clustIt = clustSet->begin();
    edmNew::DetSet<SiStripCluster>::const_iterator end = clustSet->end();

    DetId detIdObject(clustSet->detId());
    edmNew::DetSetVector<SiStripCluster>::FastFiller spc(*output, detIdObject.rawId());
    const StripGeomDetUnit* theGeomDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet(detIdObject));
    const StripTopology* topol = dynamic_cast<const StripTopology*>(&(theGeomDet->specificTopology()));

    for (; clustIt != end; ++clustIt) {
      LocalPoint lpclust = topol->localPosition(clustIt->barycenter());
      GlobalPoint GPclust = theGeomDet->surface().toGlobal(Local3DPoint(lpclust.x(), lpclust.y(), lpclust.z()));
      double clustY = GPclust.y();
      if ((clustY * y_) > 0) {
        spc.push_back(*clustIt);
      }
    }
  }
  event.put(std::move(output));
}

DEFINE_FWK_MODULE(StripClusterSelectorTopBottom);
