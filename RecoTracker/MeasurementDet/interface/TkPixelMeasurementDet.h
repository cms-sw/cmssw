#ifndef TkPixelMeasurementDet_H
#define TkPixelMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

class TransientTrackingRecHit;

class TkPixelMeasurementDet : public MeasurementDet {
public:

  typedef SiPixelClusterCollection::Range                ClusterRange;
  typedef SiPixelClusterCollection::ContainerIterator    ClusterIterator;
  typedef PixelClusterParameterEstimator::LocalValues    LocalValues;

  TkPixelMeasurementDet( const GeomDet* gdet,
			 const PixelClusterParameterEstimator* cpe);

  void update( const ClusterRange& range) { theClusterRange = range;}

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const;

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;

  const PixelGeomDetUnit& specificGeomDet() const {return *thePixelGDU;}

  TransientTrackingRecHit* buildRecHit( const SiPixelCluster& cluster) const;

private:

  const PixelGeomDetUnit*               thePixelGDU;
  const PixelClusterParameterEstimator* theCPE;
  ClusterRange                          theClusterRange;

};

#endif
