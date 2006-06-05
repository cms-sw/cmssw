#ifndef TkPixelMeasurementDet_H
#define TkPixelMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterFwd.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/Framework/interface/Handle.h"

class TransientTrackingRecHit;
class LocalTrajectoryParameters;

class TkPixelMeasurementDet : public MeasurementDet {
public:

  typedef SiPixelClusterCollection::detset detset;
  typedef detset::const_iterator const_iterator;
  typedef PixelClusterParameterEstimator::LocalValues    LocalValues;

  TkPixelMeasurementDet( const GeomDet* gdet,
			 const PixelClusterParameterEstimator* cpe);

  void update( const detset & detSet, 
	       const edm::Handle<SiPixelClusterCollection> h,
	       unsigned int id ) { 
    detSet_ = & detSet; 
    handle_ = h;
    id_ = id;
  }

  virtual ~TkPixelMeasurementDet() { }

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface& ) const;

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;

  const PixelGeomDetUnit& specificGeomDet() const {return *thePixelGDU;}

  TransientTrackingRecHit* buildRecHit( const SiPixelClusterRef & cluster,
					const LocalTrajectoryParameters & ltp) const;

private:

  const PixelGeomDetUnit*               thePixelGDU;
  const PixelClusterParameterEstimator* theCPE;
  const detset * detSet_;
  edm::Handle<SiPixelClusterCollection> handle_;
  unsigned int id_;
};

#endif
