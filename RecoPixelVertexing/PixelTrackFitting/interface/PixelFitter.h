#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitter_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitter_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"

#include <memory>

class PixelFitter {
public:
  PixelFitter() {}
  explicit PixelFitter(std::unique_ptr<PixelFitterBase> fitter): fitter_(std::move(fitter)) {}

  void swap(PixelFitter& o) { std::swap(fitter_, o.fitter_); }

  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits, const TrackingRegion& region) const {
    return fitter_->run(hits, region);
  }

private:
  std::unique_ptr<PixelFitterBase> fitter_;
};

#endif
