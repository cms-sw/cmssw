#ifndef DATAFORMATS_HCALRECHIT_HFQIE10INFO_H
#define DATAFORMATS_HCALRECHIT_HFQIE10INFO_H

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

/** \class HFQIE10Info
*
* Class to contain the info needed to perform HF reconstruction
* using QIE10 chips and dual-anode readout. Intended for use
* inside HFPreRecHit.
*/
class HFQIE10Info
{
public:
    typedef HcalDetId key_type;

    HFQIE10Info();

    HFQIE10Info(const HcalDetId& id, int adc, float charge,
                float energy, float timeRising, float timeFalling);

    inline HcalDetId id() const {return id_;}

    inline int adc() const {return adc_;}
    inline float charge() const {return charge_;}
    inline float energy() const {return energy_;}
    inline float timeRising() const {return timeRising_;}
    inline float timeFalling() const {return timeFalling_;}

private:
    HcalDetId id_;

    int32_t adc_;
    float charge_;
    float energy_;
    float timeRising_;
    float timeFalling_;
};

#endif // DATAFORMATS_HCALRECHIT_HFQIE10INFO_H
