#ifndef KFBasedPixelFitter_H
#define KFBasedPixelFitter_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>

namespace edm {class ParameterSet; class EventSetup;}
namespace reco { class Track;}

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
      const edm::EventSetup& es,
      const std::vector<const TrackingRecHit *>& hits,
      const TrackingRegion& region) const;
private:

  std::string thePropagatorLabel;
  std::string thePropagatorOppositeLabel;
  bool theUseBeamSpot; 
  edm::InputTag theBeamSpot;
  std::string theTTRHBuilderName;
  

};
#endif
