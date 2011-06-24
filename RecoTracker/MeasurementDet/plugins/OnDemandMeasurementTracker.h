#ifndef RecoTracker_MeasurementDet_OnDemandMeasurementTracker_H
#define RecoTracker_MeasurementDet_OnDemandMeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDetSystem.h"
#include "MeasurementTrackerImpl.h"
#include "DataFormats/Common/interface/RefGetter.h" 
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

#include "TkStripMeasurementDet.h"

 
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
 
  /// MeasurementTracker overloaded function
  void update( const edm::Event&) const;
  void updateStrips( const edm::Event& event) const;

  typedef edm::LazyGetter<SiStripCluster> LazyGetter;
  typedef edm::RefGetter<SiStripCluster> RefGetter;

  /// OnDemandMeasurementTracker specific function to be called to define the region in the RefGetter according to MeasurementDet content
  void define(const edm::Handle< edm::LazyGetter<SiStripCluster> > & ,
	      std::auto_ptr< RefGetter > &  ) const;

  /// MeasurementDetSystem interface
  virtual const MeasurementDet*       idToDet(const DetId& id) const;
    
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
  
  /// the handle is retrieved from the event to make reference to cluster in it
  mutable edm::Handle< edm::RefGetter<SiStripCluster> > theRefGetterH;
  mutable edm::Handle< edm::LazyGetter<SiStripCluster> > theLazyGetterH;
  mutable bool theSkipClusterRefs;
  mutable edm::Handle< edmNew::DetSetVector<TkStripMeasurementDet::SiStripRegionalClusterRef> > theStripClusterRefs;
  /// a class that holds flags, region_range (in RefGetter) for a given MeasurementDet
  class DetODStatus {
  public:
    DetODStatus(MeasurementDet * m):defined(false),updated(false),mdet(m){ region_range = std::pair<unsigned int,unsigned int>(0,0);}
      bool defined;
      bool updated;
      std::pair<unsigned int, unsigned int> region_range;
      MeasurementDet * mdet;
  };

  typedef std::map<DetId, DetODStatus> DetODContainer;
  /// mapping of detid -> MeasurementDet+flags+region_range
  mutable DetODContainer theDetODMap;

  /// mapping of elementIndex -> iterator to the DetODMap: to know what are the regions that needs to be defined in the ref getter
  mutable std::map<SiStripRegionCabling::ElementIndex, std::vector< DetODContainer::iterator> > region_mapping;

  /// assigne the cluster iterator to the TkStipMeasurementDet (const_cast in the way)
    void assign(const  TkStripMeasurementDet * csmdet,
	      DetODContainer::iterator * alreadyFound=0) const;

  /// some printouts, exclusively under LogDebug
  std::string dumpCluster(const std::vector<SiStripCluster> ::const_iterator & begin, const  std::vector<SiStripCluster> ::const_iterator& end)const;
  std::string dumpRegion(std::pair<unsigned int,unsigned int> indexes,
			 const RefGetter & theGetter,
			 bool stayUnpacked = false)const;

  mutable std::vector<uint32_t> theRawInactiveStripDetIds;
      
};

#endif
