#include <algorithm>
#include <type_traits>

#include "DataFormats/HcalRecHit/interface/CaloRecHitAuxSetter.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HFAnodeStatus.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFRecHitAuxSetter.h"

void HFRecHitAuxSetter::setAux(const HFPreRecHit& prehit,
                               const unsigned anodeStates[2],
                               const unsigned u_soiPhase,
                               HFRecHit* rechit) {
  using namespace CaloRecHitAuxSetter;

  const int wantedPhase = u_soiPhase < 2U ? u_soiPhase : 2U;

  for (unsigned ianode = 0; ianode < 2; ++ianode) {
    uint32_t aux = 0;
    const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
    if (anodeInfo) {
      const int nRaw = anodeInfo->nRaw();
      const int soiStored = anodeInfo->soi();
      if (soiStored < nRaw) {
        // SOI is in the raw data. Figure out a good
        // way to map ADCs into the three bytes available.
        int shift = 0;
        int nStore = nRaw;
        if (nRaw > 3) {
          nStore = 3;
          if (soiStored > wantedPhase)
            shift = soiStored - wantedPhase;
          if (shift + nStore > nRaw)
            shift = nRaw - nStore;
        }

        // Fill out the ADC fields
        for (int iadc = 0; iadc < nStore; ++iadc) {
          const int istored = iadc + shift;
          const QIE10DataFrame::Sample ts(anodeInfo->getRaw(istored));
          setField(&aux, 0xff, iadc * 8, ts.adc());
          if (istored == soiStored)
            setField(&aux, MASK_CAPID, OFF_CAPID, ts.capid());
        }
        setField(&aux, MASK_SOI, OFF_SOI, soiStored - shift);
      } else
        setField(&aux, MASK_SOI, OFF_SOI, 3U);
    }

    // Remember anode status
    static_assert(HFAnodeStatus::N_POSSIBLE_STATES <= MASK_STATUS + 1,
                  "Possible states enum must fit into the bit field");
    setField(&aux, MASK_STATUS, OFF_STATUS, anodeStates[ianode]);

    // Fill the aux field in the rechit
    if (ianode)
      rechit->setAuxHF(aux);
    else
      rechit->setAux(aux);
  }
}
