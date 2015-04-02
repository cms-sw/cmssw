#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitProduct_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitProduct_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"

#include <memory>
#include <vector>

class TrackingRecHitProduct
{
    protected:
        const DetId& _detId;
        
        std::vector<const PSimHit*> _simHits;
        
        std::vector<SiTrackerGSRecHit2D> _recHits;
        std::vector<SiTrackerGSMatchedRecHit2D> _matchedRecHits;
        
    public:
        TrackingRecHitProduct(const DetId& detId, std::vector<const PSimHit*> simHits):
            _detId(detId),
            _simHits(simHits)
        {
        }
        
        inline const DetId& getDetId() const
        {
            return _detId;
        }
        
        virtual std::vector<const PSimHit*>& getSimHits()
        {
            return _simHits;
        }
        
        virtual std::vector<SiTrackerGSRecHit2D>& getRecHits()
        {
            return _recHits;
        }
        
        virtual std::vector<SiTrackerGSMatchedRecHit2D>& getMatchedRecHits()
        {
            return _matchedRecHits;
        }
        
        virtual ~TrackingRecHitProduct()
        {
        }
};

typedef std::shared_ptr<TrackingRecHitProduct> TrackingRecHitProductPtr;

#endif

