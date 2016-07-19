#include "RecoLocalCalo/HcalRecAlgos/interface/HBHERecHitAuxSetter.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/CaloRecHitAuxSetter.h"

void HBHERecHitAuxSetter::setAux(const HBHEChannelInfo& info,
                                 HBHERecHit* rechit)
{
    using namespace CaloRecHitAuxSetter;

    uint32_t aux = 0, auxHBHE = 0, auxPhase1 = 0;

    // Pack ADC values
    unsigned nSamples = info.nSamples();
    if (nSamples > 10)
        nSamples = 10;
    for (unsigned i=0; i<4 && i<nSamples; ++i)
        setField(&aux, 0xff, i*8, info.tsAdc(i));
    for (unsigned i=4; i<8 && i<nSamples; ++i)
        setField(&auxHBHE, 0xff, (i-4)*8, info.tsAdc(i));
    for (unsigned i=8; i<nSamples; ++i)
        setField(&auxPhase1, 0xff, (i-8)*8, info.tsAdc(i));

    // Pack other fields
    setField(&auxPhase1, MASK_NSAMPLES, OFF_NSAMPLES, nSamples);
    unsigned soi = info.soi();
    if (soi > 10)
        soi = 10;
    setField(&auxPhase1, MASK_SOI, OFF_SOI, soi);
    setField(&auxPhase1, MASK_CAPID, OFF_CAPID, info.capid());

    // Pack status bits
    setBit(&auxPhase1, OFF_TDC_TIME, info.hasTimeInfo());
    setBit(&auxPhase1, OFF_DROPPED, info.isDropped());
    setBit(&auxPhase1, OFF_LINK_ERR, info.hasLinkError());
    setBit(&auxPhase1, OFF_CAPID_ERR, info.hasCapidError());

    // Copy the aux words into the rechit
    rechit->setAux(aux);
    rechit->setAuxHBHE(auxHBHE);
    rechit->setAuxPhase1(auxPhase1);
}
