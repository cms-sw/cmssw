#include <type_traits>

#include "RecoLocalCalo/HcalRecAlgos/interface/HFAnodeStatus.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFRecHitAuxSetter.h"

void HFRecHitAuxSetter::setAux(const HFPreRecHit& prehit,
                               const unsigned anodeStates[2],
                               HFRecHit* rechit)
{
    for (unsigned ianode=0; ianode<2; ++ianode)
    {
        unsigned aux = 0;
        const HFQIE10Info* anodeInfo = prehit.getHFQIE10Info(ianode);
        if (anodeInfo)
        {
            const int nRaw = anodeInfo->nRaw();
            if (nRaw)
            {
                int nAdc = nRaw;
                if (nAdc > 3)
                    nAdc = 3;
                int ifrom = nRaw - 1;

                const QIE10DataFrame::Sample s(anodeInfo->getRaw(ifrom));
                setField(&aux, MASK_CAPID, OFF_CAPID, s.capid());
                setField(&aux, MASK_NTS, OFF_NTS, nAdc);

                int ito = nAdc - 1;
                for ( ; ifrom >= 0 && ito >= 0; --ifrom, --ito)
                {
                    const QIE10DataFrame::Sample ts(anodeInfo->getRaw(ifrom));
                    setField(&aux, 0xff, ito*8, ts.adc());
                }
            }
        }
        static_assert(HFAnodeStatus::N_POSSIBLE_STATES <= MASK_STATUS+1,
                      "Possible states enum must fit into the bit field");
        setField(&aux, MASK_STATUS, OFF_STATUS, anodeStates[ianode]);

        if (ianode)
            rechit->setAuxHF(aux);
        else
            rechit->setAux(aux);
    }
}
