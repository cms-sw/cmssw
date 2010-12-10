#ifndef TkPixelMeasurementDet_H
#define TkPixelMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
//#include "DataFormats/SiPixelCluster/interface/SiPixelClusterFwd.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class TransientTrackingRecHit;
class LocalTrajectoryParameters;

class TkPixelMeasurementDet : public MeasurementDet {
public:

  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> SiPixelClusterRef;
  
  typedef edmNew::DetSet<SiPixelCluster> detset;
  typedef detset::const_iterator const_iterator;
  typedef PixelClusterParameterEstimator::LocalValues    LocalValues;

  TkPixelMeasurementDet( const GeomDet* gdet,
			 const PixelClusterParameterEstimator* cpe);

  void update( const detset & detSet, 
	       const edm::Handle<edmNew::DetSetVector<SiPixelCluster> > h,
	       unsigned int id ) { 
    detSet_ = detSet; 
    handle_ = h;
    id_ = id;
    empty = false;
    activeThisEvent_ = true;
  }
  void setEmpty(){empty = true; activeThisEvent_ = true; }

  virtual ~TkPixelMeasurementDet() { }

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface& ) const;

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;

  const PixelGeomDetUnit& specificGeomDet() const {return *thePixelGDU;}

  TransientTrackingRecHit::RecHitPointer 
  buildRecHit( const SiPixelClusterRef & cluster,
	       const LocalTrajectoryParameters & ltp) const;

  /** \brief Turn on/off the module for reconstruction, for the full run or lumi (using info from DB, usually).
             This also resets the 'setActiveThisEvent' to true */
  void setActive(bool active) { activeThisPeriod_ = active; activeThisEvent_ = true; if (!active) empty = true; }
  /** \brief Turn on/off the module for reconstruction for one events.
             This per-event flag is cleared by any call to 'update' or 'setEmpty'  */
  void setActiveThisEvent(bool active) { activeThisEvent_ = active;  if (!active) empty = true; }
  /** \brief Is this module active in reconstruction? It must be both 'setActiveThisEvent' and 'setActive'. */
  bool isActive() const { return activeThisEvent_ && activeThisPeriod_; }

  bool hasBadComponents( const TrajectoryStateOnSurface &tsos ) const ; 

  /** \brief Sets the list of bad ROCs, identified by the positions of their centers in the local coordinate frame*/
  void setBadRocPositions(std::vector< LocalPoint > & positions) { badRocPositions_.swap(positions); }
  /** \brief Clear the list of bad ROCs */
  void clearBadRocPositions() { badRocPositions_.clear(); }
private:

  const PixelGeomDetUnit*               thePixelGDU;
  const PixelClusterParameterEstimator* theCPE;
  detset detSet_;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > handle_;
  unsigned int id_;
  bool empty;
  bool activeThisEvent_, activeThisPeriod_;
  std::vector< LocalPoint > badRocPositions_;
  
  static const float theRocWidth, theRocHeight;

  std::set<SiPixelClusterRef> skipClusters_;
 public:
  template <typename IT>
    void setClusterToSkip(IT begin, IT end){
    skipClusters_.clear();
    skipClusters_.insert(begin,end);
      }
};

#endif
