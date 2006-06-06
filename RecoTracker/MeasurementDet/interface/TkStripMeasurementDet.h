#ifndef TkStripMeasurementDet_H
#define TkStripMeasurementDet_H

#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/Handle.h"

class TransientTrackingRecHit;

class TkStripMeasurementDet : public MeasurementDet {
public:

  //  typedef SiStripClusterCollection::Range                ClusterRange;
  //  typedef SiStripClusterCollection::ContainerIterator    ClusterIterator;
  typedef StripClusterParameterEstimator::LocalValues    LocalValues;

  typedef edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, 
    edm::refhelper::FindForDetSetVector<SiStripCluster> > SiStripClusterRef;
  
  typedef edm::DetSetVector<SiStripCluster>::detset detset;
  typedef detset::const_iterator const_iterator;

  virtual ~TkStripMeasurementDet(){}

  TkStripMeasurementDet( const GeomDet* gdet,
			 const StripClusterParameterEstimator* cpe);

  void update( const detset & detSet, 
	       const edm::Handle<edm::DetSetVector<SiStripCluster> > h,
	       unsigned int id ) { 
    detSet_ = & detSet; 
    handle_ = h;
    id_ = id;
    empty = false;
  }
  void setEmpty(){empty = true;}

  virtual RecHitContainer recHits( const TrajectoryStateOnSurface&) const;

  virtual std::vector<TrajectoryMeasurement> 
  fastMeasurements( const TrajectoryStateOnSurface& stateOnThisDet, 
		    const TrajectoryStateOnSurface& startingState, 
		    const Propagator&, 
		    const MeasurementEstimator&) const;

  const StripGeomDetUnit& specificGeomDet() const {return *theStripGDU;}

  TransientTrackingRecHit* buildRecHit( const SiStripClusterRef&,
					const LocalTrajectoryParameters& ltp) const;

private:

  const StripGeomDetUnit*               theStripGDU;
  const StripClusterParameterEstimator* theCPE;
  const detset * detSet_;
  edm::Handle<edm::DetSetVector<SiStripCluster> > handle_;
  unsigned int id_;
  bool empty;

};

#endif
