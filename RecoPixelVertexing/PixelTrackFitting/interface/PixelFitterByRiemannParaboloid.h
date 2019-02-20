#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitterByRiemannParaboloid_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitterByRiemannParaboloid_H

#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

class PixelFitterByRiemannParaboloid final : public PixelFitterBase {
public:
  explicit PixelFitterByRiemannParaboloid(const MagneticField *field,
      bool useErrors, bool useMultipleScattering);
  ~PixelFitterByRiemannParaboloid() override = default;
  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits,
                                   const TrackingRegion& region,
                                   const edm::EventSetup& setup) const override;

private:
  const MagneticField *field_;
  bool useErrors_;
  bool useMultipleScattering_;
};

#endif // RecoPixelVertexing_PixelTrackFitting_PixelFitterByRiemannParaboloid_H
