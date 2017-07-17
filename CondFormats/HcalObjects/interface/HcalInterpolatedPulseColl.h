#ifndef CondFormats_HcalObjects_HcalInterpolatedPulseColl_h_
#define CondFormats_HcalObjects_HcalInterpolatedPulseColl_h_

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulse.h"
#include "CondFormats/HcalObjects/interface/HBHEChannelGroups.h"

class HcalInterpolatedPulseColl
{
public:
    HcalInterpolatedPulseColl(
        const std::vector<HcalInterpolatedPulse>& pulses,
        const HBHEChannelGroups& groups);

    // Get the pulse from channel HcalDetId
    const HcalInterpolatedPulse& getChannelPulse(const HcalDetId& id) const;

    // Get the pulse by linearized HCAL channel number
    inline const HcalInterpolatedPulse& getChannelPulse(const unsigned i) const
        {return pulses_[groups_.getGroup(i)];}

    inline bool operator==(const HcalInterpolatedPulseColl& r)
        {return pulses_ == r.pulses_ && groups_ == r.groups_;}

    inline bool operator!=(const HcalInterpolatedPulseColl& r)
        {return !(*this == r);}

private:
    std::vector<HcalInterpolatedPulse> pulses_;
    HBHEChannelGroups groups_;

public:
    // Default constructor needed for serialization.
    // Do not use in application code.
    inline HcalInterpolatedPulseColl() {}

private:
    friend class boost::serialization::access;

    template<class Archive>
    inline void serialize(Archive & ar, unsigned /* version */)
    {
        ar & pulses_ & groups_;
    }
};

BOOST_CLASS_VERSION(HcalInterpolatedPulseColl, 1)

#endif // CondFormats_HcalObjects_HcalInterpolatedPulseColl_h_
