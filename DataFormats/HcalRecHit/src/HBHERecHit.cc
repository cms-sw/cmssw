#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHitAuxSetter.h"
#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"

HBHERecHit::HBHERecHit()
    : CaloRecHit(),
      chiSquared_(-1),
      rawEnergy_(-1.0e21),
      auxEnergy_(-1.0e21),
      auxHBHE_(0),
      auxPhase1_(0)
{
}

HBHERecHit::HBHERecHit(const HcalDetId& id, float energy, float timeRising, float timeFalling)
    : CaloRecHit(id,energy,timeRising),
      timeFalling_(timeFalling),
      chiSquared_(-1),
      rawEnergy_(-1.0e21),
      auxEnergy_(-1.0e21),
      auxHBHE_(0),
      auxPhase1_(0)
{
}

std::ostream& operator<<(std::ostream& s, const HBHERecHit& hit) {
  s << hit.id() << ": " << hit.energy() << " GeV";
  if (hit.eraw() > -0.9e21) {
    s << ", eraw=" << hit.eraw() << " GeV";
  }
  if (hit.eaux() > -0.9e21) {
    s << ", eaux=" << hit.eaux() << " GeV";
  }
  if(hit.time() > -998) {
    s << ", t= " << hit.time() << " to " << hit.timeFalling() << " ns";
  }
  return s;
}

bool HBHERecHit::isMerged() const
{
    return auxPhase1_ & (1U << HBHERecHitAuxSetter::OFF_COMBINED);
}

void HBHERecHit::getMergedIds(std::vector<HcalDetId>* ids) const
{
    if (ids)
    {
        ids->clear();
        if (auxPhase1_ & (1U << HBHERecHitAuxSetter::OFF_COMBINED))
        {
            const unsigned nMerged = CaloRecHitAuxSetter::getField(
                auxPhase1_, HBHERecHitAuxSetter::MASK_NSAMPLES,
                HBHERecHitAuxSetter::OFF_NSAMPLES);
            ids->reserve(nMerged);
            const HcalDetId myId(id());
            for (unsigned i=0; i<nMerged; ++i)
            {
                const unsigned depth = CaloRecHitAuxSetter::getField(auxHBHE_, 0xf, i*4);
                ids->emplace_back(myId.subdet(), myId.ieta(), myId.iphi(), depth);
            }
        }
    }
}


HcalDetId HBHERecHit::idFront() const {
  if (auxPhase1_ & (1U << HBHERecHitAuxSetter::OFF_COMBINED)) {
    const HcalDetId myId(id());
    return HcalDetId(myId.subdet(), myId.ieta(), myId.iphi(), auxHBHE_ & 0xf);
  } else {
    return id();
  }
}
