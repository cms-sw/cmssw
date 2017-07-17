#include <cassert>
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HBHELinearMap.h"

void HBHELinearMap::getChannelTriple(const unsigned index, unsigned* depth,
                                     int* ieta, unsigned* iphi) const
{
    if (index >= ChannelCount)
        throw cms::Exception("In HBHELinearMap::getChannelTriple: "
                                    "input index out of range");
    const HBHEChannelId& id = lookup_[index];
    if (depth)
        *depth = id.depth();
    if (ieta)
        *ieta = id.ieta();
    if (iphi)
        *iphi = id.iphi();
}

unsigned HBHELinearMap::find(const unsigned depth, const int ieta,
                             const unsigned iphi) const
{
    const HBHEChannelId id(depth, ieta, iphi);
    const unsigned loc = std::lower_bound(
        inverse_.begin(), inverse_.end(), MapPair(id, 0U)) - inverse_.begin();
    if (loc < ChannelCount)
        if (inverse_[loc].first == id)
            return inverse_[loc].second;
    return ChannelCount;
}

bool HBHELinearMap::isValidTriple(const unsigned depth, const int ieta,
                                  const unsigned iphi) const
{
    const unsigned ind = find(depth, ieta, iphi);
    return ind < ChannelCount;
}

unsigned HBHELinearMap::linearIndex(const unsigned depth, const int ieta,
                                    const unsigned iphi) const
{
    const unsigned ind = find(depth, ieta, iphi);
    if (ind >= ChannelCount)
        throw cms::Exception("In HBHELinearMap::linearIndex: "
                                    "invalid channel triple");
    return ind;
}

HBHELinearMap::HBHELinearMap()
{
    unsigned l = 0;
    unsigned depth = 1;

    for (int ieta = -29; ieta <= -21; ++ieta)
        for (unsigned iphi=1; iphi<72; iphi+=2)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = -20; ieta <= 20; ++ieta)
        if (ieta)
            for (unsigned iphi=1; iphi<=72; ++iphi)
                lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = 21; ieta <= 29; ++ieta)
        for (unsigned iphi=1; iphi<72; iphi+=2)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    depth = 2;

    for (int ieta = -29; ieta <= -21; ++ieta)
        for (unsigned iphi=1; iphi<72; iphi+=2)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = -20; ieta <= -18; ++ieta)
        for (unsigned iphi=1; iphi<=72; ++iphi)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = -16; ieta <= -15; ++ieta)
        for (unsigned iphi=1; iphi<=72; ++iphi)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = 15; ieta <= 16; ++ieta)
        for (unsigned iphi=1; iphi<=72; ++iphi)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = 18; ieta <= 20; ++ieta)
        for (unsigned iphi=1; iphi<=72; ++iphi)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = 21; ieta <= 29; ++ieta)
        for (unsigned iphi=1; iphi<72; iphi+=2)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    depth = 3;

    for (int ieta = -28; ieta <= -27; ++ieta)
        for (unsigned iphi=1; iphi<72; iphi+=2)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = -16; ieta <= -16; ++ieta)
        for (unsigned iphi=1; iphi<=72; ++iphi)
             lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = 16; ieta <= 16; ++ieta)
        for (unsigned iphi=1; iphi<=72; ++iphi)
             lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    for (int ieta = 27; ieta <= 28; ++ieta)
        for (unsigned iphi=1; iphi<72; iphi+=2)
            lookup_[l++] = HBHEChannelId(depth, ieta, iphi);

    assert(l == ChannelCount);

    inverse_.reserve(ChannelCount);
    for (unsigned i=0; i<ChannelCount; ++i)
        inverse_.push_back(MapPair(lookup_[i], i));
    std::sort(inverse_.begin(), inverse_.end());
}

HcalSubdetector HBHELinearMap::getSubdetector(const unsigned depth,
                                              const int ieta)
{
    const unsigned abseta = std::abs(ieta);

    // Make sure the arguments are in range
    if (!(abseta <= 29U))
        throw cms::Exception("In HBHELinearMap::getSubdetector: "
                                    "eta argument out of range");
    if (!(depth > 0U && depth < 4U))
        throw cms::Exception("In HBHELinearMap::getSubdetector: "
                                    "depth argument out of range");
    if (abseta == 29U)
        if (!(depth <= 2U))
            throw cms::Exception("In HBHELinearMap::getSubdetector: "
                                        "depth argument out of range "
                                        "for |ieta| = 29");
    if (abseta <= 15U)
        return HcalBarrel;
    else if (abseta == 16U)
        return depth <= 2U ? HcalBarrel : HcalEndcap;
    else
        return HcalEndcap;
}

const HBHELinearMap& hbheChannelMap()
{
    static const HBHELinearMap chMap;
    return chMap;
}
