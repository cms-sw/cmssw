#ifndef DATAFORMATS_HCALRECHIT_HFQIE10INFO_H
#define DATAFORMATS_HCALRECHIT_HFQIE10INFO_H

#include <limits>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"

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
    typedef QIE10DataFrame::Sample::wide_type raw_type;

    static const unsigned N_RAW_MAX = 5;
    static const raw_type INVALID_RAW = std::numeric_limits<raw_type>::max();

    HFQIE10Info();

    // Argument "soi" provides the index of the sample of interest
    // in the "rawData" array
    HFQIE10Info(const HcalDetId& id, float charge, float energy,
                float timeRising, float timeFalling,
                const raw_type* rawData, unsigned nData, unsigned soi);

    inline HcalDetId id() const {return id_;}

    inline float charge() const {return charge_;}
    inline float energy() const {return energy_;}
    inline float timeRising() const {return timeRising_;}
    inline float timeFalling() const {return timeFalling_;}
    inline unsigned nRaw() const {return nRaw_;}
    inline unsigned soi() const {return soi_;}
    inline raw_type getRaw(const unsigned which) const
       {return which >= nRaw_ ? INVALID_RAW : raw_[which];}

    // Check whether the "ok" flag is set in the dataframe.
    //
    // If "checkAllTimeSlices" is "true" or if the raw data
    // does not contain the "sample of interest" time slice,
    // we are going to check all time slices. Otherwise only
    // the "sample of interest" time slice is checked.
    //
    bool isDataframeOK(bool checkAllTimeSlices = false) const;

private:
    HcalDetId id_;

    float charge_;
    float energy_;
    float timeRising_;
    float timeFalling_;
    raw_type raw_[N_RAW_MAX];
    uint8_t nRaw_;
    uint8_t soi_;
};

#endif // DATAFORMATS_HCALRECHIT_HFQIE10INFO_H
