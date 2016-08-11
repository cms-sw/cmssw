#ifndef DATAFORMATS_HCALRECHIT_HFPRERECHIT_H
#define DATAFORMATS_HCALRECHIT_HFPRERECHIT_H

#include <utility>

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

    // Quantities simply added from both anodes
    float charge() const;
    float energy() const;

    // The following function returns a pair.
    // The first element of the pair is the charge asymmetry,
    // if calculated. The second element of the pair indicates
    // whether the asymmetry is actually calculated. "true"
    // means yes, it is. The asymmetry is calculated if all
    // of the following conditions are satisfied:
    //
    //   1) The data is available for both PMT anodes.
    //   2) The sum of the charges is positive.
    //   3) The sum of the charges is at least "chargeThreshold".
    //
    // If the asymmetry is not calculated, the first element of
    // the pair is set to 0 and the second to "false".
    //
    std::pair<float,bool> chargeAsymmetry(float chargeThreshold) const;

    // Similar function for the energy asymmetry
    std::pair<float,bool> energyAsymmetry(float energyThreshold) const;

private:
    HcalDetId id_;

    HFQIE10Info hfQIE10Info_[2];
    bool hasInfo_[2];
};

#endif // DATAFORMATS_HCALRECHIT_HFPRERECHIT_H
