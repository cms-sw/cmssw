#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitPipe_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitPipe_H

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include <vector>

class TrackingRecHitPipe {
protected:
  std::vector<TrackingRecHitAlgorithm*> _algorithms;

public:
  TrackingRecHitPipe() {}

  TrackingRecHitProductPtr produce(TrackingRecHitProductPtr product) const {
    for (unsigned int ialgo = 0; product && (ialgo < _algorithms.size()); ++ialgo) {
      product = _algorithms[ialgo]->process(product);
    }
    return product;
  }

  inline unsigned int size() const { return _algorithms.size(); }

  inline void addAlgorithm(TrackingRecHitAlgorithm* algorithm) { _algorithms.push_back(algorithm); }
};

#endif
