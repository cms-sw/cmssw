#ifndef RecoMTD_TransientTrackingRecHit_MTDTransientTrackingRecHit_h
#define RecoMTD_TransientTrackingRecHit_MTDTransientTrackingRecHit_h

/** \class MTDTransientTrackingRecHit
 *
 *  A TransientTrackingRecHit for MTD.
 *
 *
 *   \author   L. Gray - FNAL
 *
 */

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class MTDTransientTrackingRecHit final : public GenericTransientTrackingRecHit {
public:
  using MTDRecHitPointer = std::shared_ptr<MTDTransientTrackingRecHit>;
  using ConstMTDRecHitPointer = std::shared_ptr<MTDTransientTrackingRecHit const>;

  typedef std::vector<MTDRecHitPointer> MTDRecHitContainer;
  typedef std::vector<ConstMTDRecHitPointer> ConstMTDRecHitContainer;

  ~MTDTransientTrackingRecHit() override {}

  /// if this rec hit is a BTL rec hit
  bool isBTL() const;

  /// if this rec hit is a ETL rec hit
  bool isETL() const;

  static RecHitPointer build(const GeomDet* geom, const TrackingRecHit* rh) {
    return RecHitPointer(new MTDTransientTrackingRecHit(geom, rh));
  }

  static MTDRecHitPointer specificBuild(const GeomDet* geom, const TrackingRecHit* rh) {
    LogDebug("MTDTransientTrackingRecHit") << "Getting specificBuild" << std::endl;
    return MTDRecHitPointer(new MTDTransientTrackingRecHit(geom, rh));
  }

  void invalidateHit();

private:
  /// Construct from a TrackingRecHit and its GeomDet
  MTDTransientTrackingRecHit(const GeomDet* geom, const TrackingRecHit* rh);

  /// Copy ctor
  MTDTransientTrackingRecHit(const MTDTransientTrackingRecHit& other);

  MTDTransientTrackingRecHit* clone() const override { return new MTDTransientTrackingRecHit(*this); }
};
#endif
