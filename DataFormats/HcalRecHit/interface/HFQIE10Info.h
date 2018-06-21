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

    constexpr HFQIE10Info()
        : charge_(0.f), energy_(0.f), timeRising_(0.f), timeFalling_(-1.f), 
        raw_{INVALID_RAW, INVALID_RAW, INVALID_RAW, INVALID_RAW, INVALID_RAW},
        nRaw_(0), soi_(0)
    {}

    // Argument "soi" provides the index of the sample of interest
    // in the "rawData" array
    constexpr HFQIE10Info(const HcalDetId& id, float charge, float energy,
                float timeRising, float timeFalling,
                const raw_type* rawData, unsigned nData, unsigned soi) 
        : id_(id), charge_(charge), energy_(energy), timeRising_(timeRising),
        timeFalling_(timeFalling), 
        raw_{INVALID_RAW, INVALID_RAW, INVALID_RAW, INVALID_RAW, INVALID_RAW},
        nRaw_(std::min(nData, N_RAW_MAX)), soi_(0)
    {
        if (nData) {
            unsigned tbegin = 0;
            if (soi >= nData)
            {
                // No SOI in the data. This situation is not normal
                // but can not be addressed in this code.
                if (nData > nRaw_)
                    tbegin = nData - nRaw_;
                soi_ = nRaw_;
            } else {
                if (nData > nRaw_) {
                    // Want to keep at least 2 presamples
                    if (soi > 2U) {
                        tbegin = soi - 2U;
                        if (tbegin + nRaw_ > nData)
                            tbegin = nData - nRaw_;
                    }
                }
                soi_ = soi - tbegin;
            }

            raw_type* to = &raw_[0];
            const raw_type* from = rawData + tbegin;
            for (unsigned i=0; i<nRaw_; ++i)
                *to++ = *from++;
        }
    }

    constexpr HcalDetId id() const {return id_;}

    constexpr float charge() const {return charge_;}
    constexpr float energy() const {return energy_;}
    constexpr float timeRising() const {return timeRising_;}
    constexpr float timeFalling() const {return timeFalling_;}
    constexpr unsigned nRaw() const {return nRaw_;}
    constexpr unsigned soi() const {return soi_;}
    constexpr raw_type getRaw(const unsigned which) const
    {return which >= nRaw_ ? INVALID_RAW : raw_[which];}

    // Check whether the "ok" flag is set in the dataframe.
    //
    // If "checkAllTimeSlices" is "true" or if the raw data
    // does not contain the "sample of interest" time slice,
    // we are going to check all time slices. Otherwise only
    // the "sample of interest" time slice is checked.
    //
    bool isDataframeOK(bool checkAllTimeSlices = false) const {
        bool hardwareOK = true;
        if (soi_ >= nRaw_ || checkAllTimeSlices)
            for (unsigned i=0; i<nRaw_ && hardwareOK; ++i) {
                const QIE10DataFrame::Sample s(raw_[i]);
                hardwareOK = s.ok();
            } else {
                const QIE10DataFrame::Sample s(raw_[soi_]);
                hardwareOK = s.ok();
            }
        return hardwareOK;
    }

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
