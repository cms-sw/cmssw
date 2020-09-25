#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

MeasurementTrackerEvent::~MeasurementTrackerEvent() {
  if (theOwner) {
    //std::cout << "Deleting owned MT @" << this << " (strip data @ " << theStripData << ")" << std::endl;
    delete theStripData;
    theStripData = nullptr;  // also sets to zero since sometimes the FWK seems
    delete thePixelData;
    thePixelData = nullptr;  // to double-delete the same object (!!!)
    delete thePhase2OTData;
    thePhase2OTData = nullptr;  // to double-delete the same object (!!!)
  }
}

MeasurementTrackerEvent::MeasurementTrackerEvent(MeasurementTrackerEvent &&other) {
  theTracker = other.theTracker;
  theStripData = other.theStripData;
  thePixelData = other.thePixelData;
  thePhase2OTData = other.thePhase2OTData;
  thePhase2OTVectorHits = other.thePhase2OTVectorHits;
  thePhase2OTVectorHitsRej = other.thePhase2OTVectorHitsRej;
  theOwner = other.theOwner;
  other.theOwner = false;  // make sure to fully transfer the ownership
  theStripClustersToSkip = std::move(other.theStripClustersToSkip);
  thePixelClustersToSkip = std::move(other.thePixelClustersToSkip);
}
MeasurementTrackerEvent &MeasurementTrackerEvent::operator=(MeasurementTrackerEvent &&other) {
  theTracker = other.theTracker;
  theStripData = other.theStripData;
  thePixelData = other.thePixelData;
  thePhase2OTData = other.thePhase2OTData;
  thePhase2OTVectorHits = other.thePhase2OTVectorHits;
  thePhase2OTVectorHitsRej = other.thePhase2OTVectorHitsRej;
  theOwner = other.theOwner;
  other.theOwner = false;  // make sure to fully transfer the ownership
  theStripClustersToSkip = std::move(other.theStripClustersToSkip);
  thePixelClustersToSkip = std::move(other.thePixelClustersToSkip);
  return *this;
}

MeasurementTrackerEvent::MeasurementTrackerEvent(
    const MeasurementTrackerEvent &trackerEvent,
    const edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > &stripClustersToSkip,
    const edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > &pixelClustersToSkip)
    : theTracker(trackerEvent.theTracker),
      theStripData(trackerEvent.theStripData),
      thePixelData(trackerEvent.thePixelData),
      thePhase2OTData(nullptr),
      thePhase2OTVectorHits(nullptr),
      thePhase2OTVectorHitsRej(nullptr),
      theOwner(false) {
  //std::cout << "Creatign non-owned MT @ " << this << " from @ " << & trackerEvent << " (strip data @ " << trackerEvent.theStripData << ")" << std::endl;
  if (stripClustersToSkip.refProd().id() != theStripData->handle().id()) {
    edm::LogError("ProductIdMismatch") << "The strip masking does not point to the strip collection of clusters: "
                                       << stripClustersToSkip.refProd().id() << "!=" << theStripData->handle().id();
    throw cms::Exception("Configuration")
        << "The strip masking does not point to the strip collection of clusters: "
        << stripClustersToSkip.refProd().id() << "!=" << theStripData->handle().id() << "\n";
  }

  if (pixelClustersToSkip.refProd().id() != thePixelData->handle().id()) {
    edm::LogError("ProductIdMismatch") << "The pixel masking does not point to the proper collection of clusters: "
                                       << pixelClustersToSkip.refProd().id() << "!=" << thePixelData->handle().id();
    throw cms::Exception("Configuration")
        << "The pixel masking does not point to the proper collection of clusters: "
        << pixelClustersToSkip.refProd().id() << "!=" << thePixelData->handle().id() << "\n";
  }

  theStripClustersToSkip.resize(stripClustersToSkip.size());
  stripClustersToSkip.copyMaskTo(theStripClustersToSkip);

  thePixelClustersToSkip.resize(pixelClustersToSkip.size());
  pixelClustersToSkip.copyMaskTo(thePixelClustersToSkip);
}

//FIXME:just temporary solution for phase2!
MeasurementTrackerEvent::MeasurementTrackerEvent(
    const MeasurementTrackerEvent &trackerEvent,
    const edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > &pixelClustersToSkip,
    const edm::ContainerMask<edmNew::DetSetVector<Phase2TrackerCluster1D> > &phase2OTClustersToSkip)
    : theTracker(trackerEvent.theTracker),
      theStripData(nullptr),
      thePixelData(trackerEvent.thePixelData),
      thePhase2OTData(trackerEvent.thePhase2OTData),
      thePhase2OTVectorHits(trackerEvent.thePhase2OTVectorHits),
      thePhase2OTVectorHitsRej(trackerEvent.thePhase2OTVectorHitsRej),
      theOwner(false) {
  if (pixelClustersToSkip.refProd().id() != thePixelData->handle().id()) {
    edm::LogError("ProductIdMismatch") << "The pixel masking does not point to the proper collection of clusters: "
                                       << pixelClustersToSkip.refProd().id() << "!=" << thePixelData->handle().id();
    throw cms::Exception("Configuration")
        << "The pixel masking does not point to the proper collection of clusters: "
        << pixelClustersToSkip.refProd().id() << "!=" << thePixelData->handle().id() << "\n";
  }

  if (phase2OTClustersToSkip.refProd().id() != thePhase2OTData->handle().id()) {
    edm::LogError("ProductIdMismatch") << "The pixel masking does not point to the proper collection of clusters: "
                                       << pixelClustersToSkip.refProd().id() << "!=" << thePixelData->handle().id();
    throw cms::Exception("Configuration")
        << "The pixel masking does not point to the proper collection of clusters: "
        << pixelClustersToSkip.refProd().id() << "!=" << thePixelData->handle().id() << "\n";
  }

  thePixelClustersToSkip.resize(pixelClustersToSkip.size());
  pixelClustersToSkip.copyMaskTo(thePixelClustersToSkip);

  thePhase2OTClustersToSkip.resize(phase2OTClustersToSkip.size());
  phase2OTClustersToSkip.copyMaskTo(thePhase2OTClustersToSkip);
}
