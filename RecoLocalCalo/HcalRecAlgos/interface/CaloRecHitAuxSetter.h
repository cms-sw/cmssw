#ifndef RecoLocalCalo_HcalRecAlgos_CaloRecHitAuxSetter_h_
#define RecoLocalCalo_HcalRecAlgos_CaloRecHitAuxSetter_h_

#include <cstdint>

// Useful helpers for uint32_t fields
namespace CaloRecHitAuxSetter
{
    inline void setField(uint32_t* u, const unsigned mask,
                         const unsigned offset, const unsigned value)
        {*u &= ~(mask << offset); *u |= ((value & mask) << offset);}

    inline unsigned getField(const uint32_t u, const unsigned mask,
                             const unsigned offset)
        {return (u >> offset) & mask;}

    inline void setBit(uint32_t* u, const unsigned bitnum, const bool b)
        {if (b) {*u |= (1U << bitnum);} else {*u &= ~(1U << bitnum);}}

    inline bool getBit(const uint32_t u, const unsigned bitnum)
        {return u & (1U << bitnum);}
}

#endif // RecoLocalCalo_HcalRecAlgos_CaloRecHitAuxSetter_h_
