#ifndef RecoLocalCalo_HcalRecAlgos_AbsPlan1RechitCombiner_h_
#define RecoLocalCalo_HcalRecAlgos_AbsPlan1RechitCombiner_h_

#include <utility>

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

class HcalTopology;

class AbsPlan1RechitCombiner {
public:
  inline virtual ~AbsPlan1RechitCombiner() {}

  // The topology should be set before the first call
  // to "add" and whenever it changes
  virtual void setTopo(const HcalTopology* topo) = 0;

  // The "clear" function is called once per event,
  // at the beginning of the rechit processing
  virtual void clear() = 0;

  // This method should be called to add a rechit to process.
  // It will be assumed that the rechit reference will remain
  // valid at the time "combine" method is called.
  virtual void add(const HBHERecHit& rh) = 0;

  // This method should be called once per event,
  // after all rechits have been added
  virtual void combine(HBHERecHitCollection* collectionToFill) = 0;

protected:
  // The first element of the pair is the value to average
  // and the second is the weight (energy). Non-positive weights
  // will be ignored.
  typedef std::pair<float, float> FPair;

  static float energyWeightedAverage(const FPair* data, unsigned len, float valueToReturnOnFailure);
};

#endif  // RecoLocalCalo_HcalRecAlgos_AbsPlan1RechitCombiner_h_
