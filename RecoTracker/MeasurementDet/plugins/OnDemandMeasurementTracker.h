#ifndef RecoTracker_MeasurementDet_OnDemandMeasurementTracker_H
#define RecoTracker_MeasurementDet_OnDemandMeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "MeasurementTrackerImpl.h"
#include "DataFormats/Common/interface/RefGetter.h" 
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "TkStripMeasurementDet.h"
#include<unordered_map>
 
class OnDemandMeasurementTracker : public MeasurementTrackerImpl {
public:
  /// constructor
  OnDemandMeasurementTracker(const edm::ParameterSet&              conf,
			     const PixelClusterParameterEstimator* pixelCPE,
			     const StripClusterParameterEstimator* stripCPE,
			     const SiStripRecHitMatcher*  hitMatcher,
			     const TrackerGeometry*  trackerGeom,
			     const GeometricSearchTracker* geometricSearchTracker,
    			     const SiStripQuality *stripQuality,
                             int   stripQualityFlags,
                             int   stripQualityDebugFlags,
                             const SiPixelQuality *pixelQuality,
                             const SiPixelFedCabling *pixelCabling,
                             int   pixelQualityFlags,
                             int   pixelQualityDebugFlags,
			     const SiStripRegionCabling * stripRegionCabling,
			     bool  isRegional=false);
  /// destructor
  virtual ~OnDemandMeasurementTracker() {}
 
  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;

  /// OnDemandMeasurementTracker specific function to be called to define the region in the RefGetter according to MeasurementDet content
  void define(const edm::Handle< edm::LazyGetter<SiStripCluster> > & ,
	      RefGetter &, StMeasurementDetSet &  ) const;

  /// MeasurementTrackerImpl interface
  const MeasurementDet * idToDetBare(const DetId& id, const MeasurementTrackerEvent &data) const ;
    
 private:
  /// log category
  std::string category_;
  /// internal flag to avoid unpacking things with LogDebug on
  bool StayPacked_;
  /// internal flag to do strip on demand (not configurable) true by default
  bool StripOnDemand_;
  /// internal flag to do pixel on demand (not configurable) false by default
  bool PixelOnDemand_;
  
  /// the cabling region tool to update a RefGetter
  const  SiStripRegionCabling * theStripRegionCabling;
  
  class DetODStatus {
  public:
    enum Kind { Pixel, Strip, Glued };
    DetODStatus(const MeasurementDet * m) : mdet(m), index(-1), kind(Pixel) {}
    const MeasurementDet * mdet;
    int index;
    Kind kind;
  };
  
  typedef std::unordered_map<unsigned int, DetODStatus> DetODContainer;
  /// mapping of detid -> MeasurementDet+flags+region_range
  DetODContainer theDetODMap;
  //int theNumberOfGluedDets;
  
  /// mapping of elementIndex -> iterator to the DetODMap: to know what are the regions that needs to be defined in the ref getter
  typedef std::vector<std::pair<SiStripRegionCabling::ElementIndex, std::vector<DetODContainer::const_iterator> > > RegionalMap;
  RegionalMap region_mapping;

  /// assigne the cluster iterator to the TkStipMeasurementDet (const_cast in the way)
    void assign(const  TkStripMeasurementDet * csmdet,
                const MeasurementTrackerEvent &data) const;

  /// some printouts, exclusively under LogDebug
  std::string dumpCluster(const std::vector<SiStripCluster> ::const_iterator & begin, const  std::vector<SiStripCluster> ::const_iterator& end)const;
  std::string dumpRegion(std::pair<unsigned int,unsigned int> indexes,
			 const RefGetter & theGetter,
			 bool stayUnpacked = false)const;

      
};

#endif
