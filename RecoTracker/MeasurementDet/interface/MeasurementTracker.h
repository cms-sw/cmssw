#ifndef MeasurementTracker_H
#define MeasurementTracker_H

#include "TrackingTools/MeasurementDet/interface/TrackingSystem.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <map>
#include <vector>

class TkStripMeasurementDet;
class TkPixelMeasurementDet;
class TrackingGeometry;

class MeasurementTracker : public TrackingSystem {
public:

  MeasurementTracker( const edm::EventSetup&, const edm::Event&);

  virtual ~MeasurementTracker() {}

  virtual const MeasurementDet* measurementDet(const DetId& id) const;
  
  void update( const edm::Event&) const;

  const TrackingGeometry* geomTracker() const { return theTrackerGeom;}

private:

  typedef std::map<DetId,MeasurementDet*>   DetContainer;

  DetContainer                        theDetMap;
  std::vector<TkStripMeasurementDet*> theStripDets;
  std::vector<TkPixelMeasurementDet*> thePixelDets;
  const TrackingGeometry*             theTrackerGeom;

  const StripClusterParameterEstimator* stripCPE;
  const PixelClusterParameterEstimator* pixelCPE;

  void initialize(const edm::EventSetup&);

  void addDet( const GeomDet* gd);
  void addStripDet( const GeomDet* gd,
		    const StripClusterParameterEstimator* cpe);
  void addPixelDet( const GeomDet* gd,
		    const PixelClusterParameterEstimator* cpe);

};

#endif
