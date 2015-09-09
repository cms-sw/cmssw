#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitProduct_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitProduct_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

#include <memory>
#include <vector>

class TrackingRecHitProduct
{
    protected:
        const DetId& _detId;
        
        std::vector<const PSimHit*> _simHits;
        
        std::vector<FastSingleTrackerRecHit> _recHits;
        std::unordered_map<unsigned int,std::vector<const PSimHit*>> _mapRecHitToSimHits;
        
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
        
        virtual void addRecHit(FastSingleTrackerRecHit & recHit, std::vector<const PSimHit*> simHits={})
        {
            _mapRecHitToSimHits[_recHits.size()]=simHits;
            _recHits.push_back(recHit);
        }

        virtual unsigned int numberOfRecHits() const
        {
            return _recHits.size();
        }

        virtual const FastSingleTrackerRecHit & getRecHit(unsigned int recHitIndex) const
        {
            return _recHits[recHitIndex];
        }
        
        virtual const std::vector<const PSimHit*>& getSimHitsFromRecHit(unsigned int recHitIndex)
        {
            return _mapRecHitToSimHits[recHitIndex];
        }
        
        virtual ~TrackingRecHitProduct()
        {
        }
};

typedef std::shared_ptr<TrackingRecHitProduct> TrackingRecHitProductPtr;

#endif

