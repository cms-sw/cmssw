#ifndef DATAFORMATS_HCALRECHIT_HFPRERECHIT_H
#define DATAFORMATS_HCALRECHIT_HFPRERECHIT_H

#include "DataFormats/HcalRecHit/interface/HFQIE10Info.h"

/** \class HFPreRecHit
*
* Class to contain the info needed to perform HF reconstruction
* using QIE10 chips and dual-anode readout. Groups the information
* provided by a single PMT.
*/
class HFPreRecHit
{
public:
    typedef HcalDetId key_type;

    HFPreRecHit();

    // In the constructor below, either "first" or "second"
    // pointer can be nullptr
    HFPreRecHit(const HcalDetId& id, const HFQIE10Info* first,
                const HFQIE10Info* second);

    inline HcalDetId id() const {return id_;}

    // Get a pointer to the QIE10 info. nullptr will be returned
    // if the info with the given index does not exist or if the
    // index is out of range.
    const HFQIE10Info* getHFQIE10Info(unsigned index) const;

private:
    HcalDetId id_;

    HFQIE10Info hfQIE10Info_[2];
    bool hasInfo_[2];
};

#endif // DATAFORMATS_HCALRECHIT_HFPRERECHIT_H
