#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

// find corresponding hypotheses based on JetLepComb
int
TtEvent::correspondingHypo(const HypoClassKey& key1, const unsigned& hyp1, const HypoClassKey& key2) const
{
  for(unsigned hyp2 = 0; hyp2 < this->numberOfAvailableHypos(key2); ++hyp2) {
    if( this->jetLepComb(key1, hyp1) == this->jetLepComb(key2, hyp2) )
      return hyp2;
  }
  return -1; // if no corresponding hypothesis was found
}
