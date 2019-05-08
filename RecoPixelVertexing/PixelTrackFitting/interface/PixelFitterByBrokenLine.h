#ifndef RecoPixelVertexing_PixelTrackFitting_PixelFitterByBrokenLine_H
#define RecoPixelVertexing_PixelTrackFitting_PixelFitterByBrokenLine_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>



class PixelFitterByBrokenLine final : public PixelFitterBase {
public:
  explicit PixelFitterByBrokenLine(const MagneticField *field);
  virtual ~PixelFitterByBrokenLine() = default;
  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits, const TrackingRegion& region, const edm::EventSetup& setup) const override;

private:
  const MagneticField *field_;
};
#endif
