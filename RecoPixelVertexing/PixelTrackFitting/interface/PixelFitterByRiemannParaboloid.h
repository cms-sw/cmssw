#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitterByRiemannParaboloid_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitterByRiemannParaboloid_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>



class PixelFitterByRiemannParaboloid final : public PixelFitterBase {
public:
  explicit PixelFitterByRiemannParaboloid(const edm::EventSetup *es, const MagneticField *field,
      bool useErrors, bool useMultipleScattering);
  virtual ~PixelFitterByRiemannParaboloid() = default;
  virtual std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits,
                                           const TrackingRegion& region) const override;

private:
  const edm::EventSetup *es_;
  const MagneticField *field_;
  bool useErrors_;
  bool useMultipleScattering_;
};
#endif
