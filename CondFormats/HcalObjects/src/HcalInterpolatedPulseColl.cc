#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseColl.h"
#include "CondFormats/HcalObjects/interface/HBHELinearMap.h"

HcalInterpolatedPulseColl::HcalInterpolatedPulseColl(
    const std::vector<HcalInterpolatedPulse>& pulses,
    const HBHEChannelGroups& groups)
    : pulses_(pulses),
      groups_(groups)
{
    if (!(pulses_.size() == groups_.largestGroupNumber() + 1U))
        throw cms::Exception(
            "Inconsistent arguments in HcalInterpolatedPulseColl constructor");
}

const HcalInterpolatedPulse& HcalInterpolatedPulseColl::getChannelPulse(
    const HcalDetId& id) const
{
    // Figure out the group number for this channel
    const unsigned lindex = hbheChannelMap().linearIndex(
        id.depth(), id.ieta(), id.iphi());
    const unsigned grN = groups_.getGroup(lindex);

    // Return the pulse for this group
    return pulses_[grN];
}
