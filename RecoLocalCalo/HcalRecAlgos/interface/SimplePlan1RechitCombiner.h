#ifndef RecoLocalCalo_HcalRecAlgos_SimplePlan1RechitCombiner_h_
#define RecoLocalCalo_HcalRecAlgos_SimplePlan1RechitCombiner_h_

// Base class header
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsPlan1RechitCombiner.h"

class SimplePlan1RechitCombiner : public AbsPlan1RechitCombiner
{
public:
    SimplePlan1RechitCombiner();

    inline ~SimplePlan1RechitCombiner() override {}

    void setTopo(const HcalTopology* topo) override;
    void clear() override;
    void add(const HBHERecHit& rh) override;
    void combine(HBHERecHitCollection* toFill) override;

protected:
    // Map the original detector id into the id of the composite rechit
    virtual HcalDetId mapRechit(HcalDetId from) const;

    // Return rechit with id of 0 if it is to be discarded
    virtual HBHERecHit makeRechit(HcalDetId idToMake,
                                  const std::vector<const HBHERecHit*>& rechits) const;

    // Combine rechit auxiliary information
    virtual void combineAuxInfo(const std::vector<const HBHERecHit*>& rechits,
                                HBHERecHit* rh) const;

    const HcalTopology* topo_;

private:
    // The first element of the pair is the id of the combined rechit.
    // The second element is the pointer to the original rechit.
    typedef std::pair<HcalDetId, const HBHERecHit*> MapItem;

    std::vector<MapItem> rechitMap_;
    std::vector<const HBHERecHit*> ptrbuf_;
};

#endif // RecoLocalCalo_HcalRecAlgos_SimplePlan1RechitCombiner_h_
