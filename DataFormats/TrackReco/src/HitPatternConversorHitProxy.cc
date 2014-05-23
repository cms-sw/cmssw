#include "DataFormats/TrackReco/interface/HitPatternConversorHitProxy.h"

using namespace reco;

DetId HitPatternConversorHitProxy::geographicalId() const
{
    return TrackingRecHit::geographicalId();
}


TrackingRecHit::Type HitPatternConversorHitProxy::type() const
{
    return TrackingRecHit::Type();
}

