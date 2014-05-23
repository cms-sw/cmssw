#ifndef __HITPATTERN_CONVERSOR_HIT_PROXY__
#define __HITPATTERN_CONVERSOR_HIT_PROXY__

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

namespace reco
{

class HitPatternConversorHitProxy : public TrackingRecHit
{
public:

    HitPatternConversorHitProxy(DetId id, TrackingRecHit::Type type) : TrackingRecHit(id, type)
    {
        ;
    };

    HitPatternConversorHitProxy(TrackingRecHit::id_type id, TrackingRecHit::Type type) : TrackingRecHit(id, type)
    {
        ;
    };

    ~HitPatternConversorHitProxy()
    {
        ;
    };

    DetId geographicalId() const;
    TrackingRecHit::Type type() const;

private:
    HitPatternConversorHitProxy() : TrackingRecHit()
    {
        ;
    };

    TrackingRecHit *clone() const
    {
        return new HitPatternConversorHitProxy();
    };

    AlgebraicVector parameters() const
    {
        return AlgebraicVector();
    };

    AlgebraicSymMatrix parametersError() const
    {
        return AlgebraicSymMatrix();
    };

    AlgebraicMatrix projectionMatrix() const
    {
        return AlgebraicMatrix();
    };

    int dimension() const
    {
        return 0;
    };

    std::vector<const TrackingRecHit*> recHits() const
    {
        return std::vector<const TrackingRecHit*>();
    };

    std::vector<TrackingRecHit*> recHits()
    {
        return std::vector<TrackingRecHit*>();
    };

    LocalPoint localPosition() const
    {
        return LocalPoint();
    };

    LocalError localPositionError() const
    {
        return LocalError();
    };
};

}

#endif

