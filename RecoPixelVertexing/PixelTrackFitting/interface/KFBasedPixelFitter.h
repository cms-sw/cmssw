#ifndef KFBasedPixelFitter_H
#define KFBasedPixelFitter_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"


namespace edm {class EventSetup;}
namespace reco { class Track; class BeamSpot; }

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;
class TrackingRegion;
class TrackingRecHit;
class Propagator;


class KFBasedPixelFitter : public PixelFitterBase {
public:
  KFBasedPixelFitter(const edm::EventSetup *es, const Propagator *propagator, const Propagator *opropagator,
                     const TransientTrackingRecHitBuilder *ttrhBuilder,
                     const TrackerGeometry *tracker, const MagneticField *field,
                     const reco::BeamSpot *beamSpot);
  virtual ~KFBasedPixelFitter() {}

  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits, const TrackingRegion& region) const override;

private:

  //this two simple classes are copied from Alignment/ReferenceTrajectories in order to avoid dependencies 
  class MyBeamSpotGeomDet final : public GeomDet {
    public:
    explicit MyBeamSpotGeomDet(const ReferenceCountingPointer<BoundPlane>& plane) :GeomDet(plane) { setDetId(0); }
    virtual ~MyBeamSpotGeomDet() { }
    virtual GeomDetEnumerators::SubDetector subDetector() const { return GeomDetEnumerators::invalidDet; }
    virtual std::vector< const GeomDet*> components() const { return std::vector< const GeomDet*>(); }
  };
  class MyBeamSpotHit final :  public TValidTrackingRecHit {
    public:
    MyBeamSpotHit (const reco::BeamSpot &beamSpot, const GeomDet * geom);
    virtual ~MyBeamSpotHit(){}
    virtual LocalPoint localPosition() const { return localPosition_; }
    virtual LocalError localPositionError() const { return localError_; }
    virtual AlgebraicVector parameters() const;
    virtual AlgebraicSymMatrix parametersError() const;
    virtual int dimension() const { return 1; }
    virtual AlgebraicMatrix projectionMatrix() const;
    virtual std::vector<const TrackingRecHit*> recHits() const { return std::vector<const TrackingRecHit*>(); }
    virtual std::vector<TrackingRecHit*> recHits() { return std::vector<TrackingRecHit*>(); }
    virtual const TrackingRecHit * hit() const { return 0; }
    private:
    LocalPoint localPosition_;
    LocalError localError_;
    virtual MyBeamSpotHit * clone() const { return new MyBeamSpotHit(*this); }
  };

  const edm::EventSetup *theES;
  const Propagator *thePropagator;
  const Propagator *theOPropagator;
  const TransientTrackingRecHitBuilder *theTTRHBuilder;
  const TrackerGeometry *theTracker;
  const MagneticField *theField;
  const reco::BeamSpot *theBeamSpot;
};
#endif
