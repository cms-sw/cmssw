#ifndef MeasurementTrackerEvent_H
#define MeasurementTrackerEvent_H

#include <vector>
class StMeasurementDetSet;
class PxMeasurementDetSet;
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/ContainerMask.h"

//// Now, to put this into the edm::Event we need a dictionary
//// and gccxml/cint can't parse the MeasurementTracker class
//// so we hide the implementation from them
#if defined(__GCCXML__) || defined(__CINT__)
    #define MeasurementTrackerEvent_Hide_Impl
    struct MeasurementTracker;
#else
    #include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#endif

class MeasurementTrackerEvent {
public:
#ifndef MeasurementTrackerEvent_Hide_Impl
   typedef MeasurementTracker::QualityFlags QualityFlags;
#endif

   /// Dummy constructor used for I/O (even if it's a transient object)
   MeasurementTrackerEvent() : theTracker(0), theStripData(0), thePixelData(0), theOwner(false), theStripClustersToSkip(), thePixelClustersToSkip() {}

   ~MeasurementTrackerEvent() ;

   /// Real constructor 1: with the full data (not owned)
   MeasurementTrackerEvent(const MeasurementTracker &tracker, const StMeasurementDetSet &strips, const PxMeasurementDetSet &pixels, 
                           const std::vector<bool> & stripClustersToSkip = std::vector<bool>(),
                           const std::vector<bool> & pixelClustersToSkip = std::vector<bool>()):
         theTracker(&tracker), theStripData(&strips), thePixelData(&pixels), theOwner(false),
         theStripClustersToSkip(stripClustersToSkip), thePixelClustersToSkip(pixelClustersToSkip) {}

   /// Real constructor 1: with the full data (owned)
   MeasurementTrackerEvent(const MeasurementTracker &tracker, const StMeasurementDetSet *strips, const PxMeasurementDetSet *pixels, 
                           const std::vector<bool> & stripClustersToSkip = std::vector<bool>(),
                           const std::vector<bool> & pixelClustersToSkip = std::vector<bool>()):
         theTracker(&tracker), theStripData(strips), thePixelData(pixels), theOwner(true),
         theStripClustersToSkip(stripClustersToSkip), thePixelClustersToSkip(pixelClustersToSkip) {}

   ///// Real constructor 2: with new cluster skips (unchecked)
   //MeasurementTrackerEvent(const MeasurementTrackerEvent &trackerEvent, 
   //                        const std::vector<bool> & stripClustersToSkip,
   //                        const std::vector<bool> & pixelClustersToSkip):
   //      theTracker(trackerEvent.theTracker), theStripData(trackerEvent.theStripData), thePixelData(trackerEvent.thePixelData), theOwner(false)
   //      theStripClustersToSkip(stripClustersToSkip), thePixelClustersToSkip(pixelClustersToSkip) {}

   /// Real constructor 2: with new cluster skips (checked)
   MeasurementTrackerEvent(const MeasurementTrackerEvent &trackerEvent, 
                           const edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > & stripClustersToSkip, 
                           const edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > & pixelClustersToSkip) ;


   MeasurementTrackerEvent(const MeasurementTrackerEvent & other) : 
        theTracker(other.theTracker), 
        theStripData(other.theStripData),
        thePixelData(other.thePixelData),
        theOwner(false),
        theStripClustersToSkip(other.theStripClustersToSkip),
        thePixelClustersToSkip(other.thePixelClustersToSkip)
   {
        assert(other.theOwner == false && "trying to copy an owning pointer"); 
   }

   MeasurementTrackerEvent & operator=(const MeasurementTrackerEvent & other)
   {
      if (&other != this) {
          MeasurementTrackerEvent copy(other);
          copy.swap(*this);
      }
      return *this;
   }

#ifndef MeasurementTrackerEvent_Hide_Impl
   MeasurementTrackerEvent(MeasurementTrackerEvent && other) : 
        theTracker(other.theTracker), 
        theStripData(other.theStripData),
        thePixelData(other.thePixelData),
        theOwner(other.theOwner),
        theStripClustersToSkip(std::move(other.theStripClustersToSkip)),
        thePixelClustersToSkip(std::move(other.thePixelClustersToSkip))
    { 
        other.theTracker = 0;
        other.theStripData = 0; other.thePixelData = 0;
        other.theOwner = false;
    }

   MeasurementTrackerEvent & operator=(MeasurementTrackerEvent && other)
   {
      if (&other != this) {
          MeasurementTrackerEvent copy(other);
          copy.swap(*this);
      }
      return *this;
   }
#endif

   void swap(MeasurementTrackerEvent &other) ;

   const MeasurementTracker & measurementTracker() const { return * theTracker; }
   const StMeasurementDetSet & stripData() const { return * theStripData; }
   const PxMeasurementDetSet & pixelData() const { return * thePixelData; }
   const std::vector<bool> & stripClustersToSkip() const { return theStripClustersToSkip; }
   const std::vector<bool> & pixelClustersToSkip() const { return thePixelClustersToSkip; }

#ifndef MeasurementTrackerEvent_Hide_Impl
   // forwarded calls
   const TrackingGeometry* geomTracker() const { return measurementTracker().geomTracker(); }
   const GeometricSearchTracker* geometricSearchTracker() const {return measurementTracker().geometricSearchTracker(); }

   /// Previous MeasurementDetSystem interface
   MeasurementDetWithData  idToDet(const DetId& id) const { return measurementTracker().idToDet(id, *this); }
#endif

protected:
   const MeasurementTracker * theTracker;
   const StMeasurementDetSet *theStripData;
   const PxMeasurementDetSet *thePixelData;
   bool  theOwner; // do I own the two above?
   // these two could be const pointers as well, but ContainerMask doesn't expose the vector
   std::vector<bool> theStripClustersToSkip; 
   std::vector<bool> thePixelClustersToSkip;
};

inline void swap(MeasurementTrackerEvent &a, MeasurementTrackerEvent &b) { a.swap(b); }
#endif // MeasurementTrackerEvent_H
