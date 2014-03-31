#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/MeasurementDet/src/TkMeasurementDetSet.h"

MeasurementTrackerEvent::~MeasurementTrackerEvent() {
    if (theOwner) {
        //std::cout << "Deleting owned MT @" << this << " (strip data @ " << theStripData << ")" << std::endl;
        delete theStripData; theStripData = 0; // also sets to zero since sometimes the FWK seems
        delete thePixelData; thePixelData = 0; // to double-delete the same object (!!!)
    }
}

void
MeasurementTrackerEvent::swap(MeasurementTrackerEvent &other)
{
    if (&other != this) {
        using std::swap;
        swap(theTracker, other.theTracker); 
        swap(theStripData, other.theStripData);
        swap(thePixelData, other.thePixelData);
        swap(theOwner, other.theOwner);
        swap(theStripClustersToSkip, other.theStripClustersToSkip);
        swap(thePixelClustersToSkip, other.thePixelClustersToSkip);
    }
}

MeasurementTrackerEvent::MeasurementTrackerEvent(const MeasurementTrackerEvent &trackerEvent,
                           const edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > & stripClustersToSkip,
                           const edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > & pixelClustersToSkip) :
     theTracker(trackerEvent.theTracker), 
     theStripData(trackerEvent.theStripData), thePixelData(trackerEvent.thePixelData), theOwner(false),
     theStripClustersToSkip(), 
     thePixelClustersToSkip() 
{
    //std::cout << "Creatign non-owned MT @ " << this << " from @ " << & trackerEvent << " (strip data @ " << trackerEvent.theStripData << ")" << std::endl;
    if (stripClustersToSkip.refProd().id() != theStripData->handle().id() ){
        edm::LogError("ProductIdMismatch")<<"The strip masking does not point to the strip collection of clusters: "<<stripClustersToSkip.refProd().id()<<"!="<<  theStripData->handle().id();
        throw cms::Exception("Configuration")<<"The strip masking does not point to the strip collection of clusters: "<<stripClustersToSkip.refProd().id()<<"!="<< theStripData->handle().id()<< "\n";
    }

    if (pixelClustersToSkip.refProd().id() != thePixelData->handle().id()){
        edm::LogError("ProductIdMismatch")<<"The pixel masking does not point to the proper collection of clusters: "<<pixelClustersToSkip.refProd().id()<<"!="<<thePixelData->handle().id();
        throw cms::Exception("Configuration")<<"The pixel masking does not point to the proper collection of clusters: "<<pixelClustersToSkip.refProd().id()<<"!="<<thePixelData->handle().id()<<"\n";
    }

    theStripClustersToSkip.resize(stripClustersToSkip.size());
    stripClustersToSkip.copyMaskTo(theStripClustersToSkip);

    thePixelClustersToSkip.resize(pixelClustersToSkip.size());
    pixelClustersToSkip.copyMaskTo(thePixelClustersToSkip);
}

