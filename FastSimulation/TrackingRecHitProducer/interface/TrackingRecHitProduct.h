#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitProduct_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitProduct_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

#include <memory>
#include <vector>
#include <unordered_map>

class TrackingRecHitProduct {
public:
  typedef std::pair<unsigned int, const PSimHit*> SimHitIdPair;
  typedef std::pair<FastSingleTrackerRecHit, std::vector<SimHitIdPair>> RecHitToSimHitIdPairs;

protected:
  const DetId& _detId;

  std::vector<SimHitIdPair> _simHitsIdPairList;

  std::vector<RecHitToSimHitIdPairs> _recHits;

public:
  TrackingRecHitProduct(const DetId& detId, std::vector<SimHitIdPair>& simHitsIdPairList)
      : _detId(detId), _simHitsIdPairList(simHitsIdPairList) {}

  inline const DetId& getDetId() const { return _detId; }

  virtual std::vector<SimHitIdPair>& getSimHitIdPairs() { return _simHitsIdPairList; }

  virtual void addRecHit(const FastSingleTrackerRecHit& recHit, std::vector<SimHitIdPair> simHitIdPairs = {}) {
    _recHits.push_back(std::make_pair(recHit, simHitIdPairs));
    for (unsigned int isimhit = 0; isimhit < simHitIdPairs.size(); ++isimhit) {
      _recHits.back().first.addSimTrackId(simHitIdPairs[isimhit].second->trackId());
    }
  }

  virtual const std::vector<RecHitToSimHitIdPairs>& getRecHitToSimHitIdPairs() const { return _recHits; }

  virtual unsigned int numberOfRecHits() const { return _recHits.size(); }

  virtual ~TrackingRecHitProduct() {}
};

typedef std::shared_ptr<TrackingRecHitProduct> TrackingRecHitProductPtr;

#endif
