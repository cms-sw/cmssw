#ifndef PixelFitterByHelixProjections_H
#define PixelFitterByHelixProjections_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>



class PixelFitterByHelixProjections final : public PixelFitterBase {
public:
  explicit PixelFitterByHelixProjections(const edm::EventSetup *es, const MagneticField *field);
  virtual ~PixelFitterByHelixProjections() {}
  virtual std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits,
                                           const TrackingRegion& region) const override;

private:
  /* these are just static and local moved to local namespace in cc .... 
   *
  int charge(const std::vector<GlobalPoint> & points) const;
  float cotTheta(const GlobalPoint& pinner, const GlobalPoint& pouter) const;
  float phi(float xC, float yC, int charge) const;
  float pt(float curvature) const;
  float zip(float d0, float phi_p, float curv, 
    const GlobalPoint& pinner, const GlobalPoint& pouter) const;
  double errZip2(float apt, float eta) const;
  double errTip2(float apt, float eta) const;
  */
private:
  const edm::EventSetup *theES;
  const MagneticField *theField;
};
#endif
