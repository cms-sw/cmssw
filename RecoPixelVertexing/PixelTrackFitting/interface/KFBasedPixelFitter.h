#ifndef KFBasedPixelFitter_H
#define KFBasedPixelFitter_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TValidTrackingRecHit.h"


namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco { class Track; class BeamSpot; }

class TransientTrackingRecHitBuilder;
class TrackerGeometry;
class MagneticField;
class TrackingRegion;
class TrackingRecHit;


class KFBasedPixelFitter : public PixelFitter {
public:
  KFBasedPixelFitter(  const edm::ParameterSet& cfg);
  virtual ~KFBasedPixelFitter() {}
    virtual reco::Track* run(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits,
      const TrackingRegion& region) const;
private:

  //this two simple classes are copied from Alignment/ReferenceTrajectories in order to avoid dependencies 
  class MyBeamSpotGeomDet GCC11_FINAL : public GeomDet {
    public:
    explicit MyBeamSpotGeomDet(const ReferenceCountingPointer<BoundPlane>& plane) :GeomDet(plane) { setDetId(0); }
    virtual ~MyBeamSpotGeomDet() { }
    virtual GeomDetEnumerators::SubDetector subDetector() const { return GeomDetEnumerators::invalidDet; }
    virtual std::vector< const GeomDet*> components() const { return std::vector< const GeomDet*>(); }
  };
  class MyBeamSpotHit GCC11_FINAL :  public TValidTrackingRecHit {
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
  

  std::string thePropagatorLabel;
  std::string thePropagatorOppositeLabel;
  bool theUseBeamSpot; 
  edm::InputTag theBeamSpot;
  std::string theTTRHBuilderName;
  

};
#endif
