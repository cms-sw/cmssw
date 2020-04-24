#ifndef CondFormats_HcalObjects_HBHELinearMap_h_
#define CondFormats_HcalObjects_HBHELinearMap_h_

//
// Linearize the channel id in the HBHE
//
// I. Volobouev
// September 2014
//

#include <vector>
#include <utility>

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class HBHELinearMap
{
public:
    enum {ChannelCount = 5184U};

    HBHELinearMap();

    // Mapping from the depth/ieta/iphi triple which uniquely
    // identifies an HBHE channel into a linear index, currently
    // from 0 to 5183 (inclusive). This linear index should not
    // be treated as anything meaningful -- consider it to be
    // just a convenient unique key in a database table.
    unsigned linearIndex(unsigned depth, int ieta, unsigned iphi) const;

    // Check whether the given triple is a valid depth/ieta/iphi combination
    bool isValidTriple(unsigned depth, int ieta, unsigned iphi) const;

    // Inverse mapping, from a linear index into depth/ieta/iphi triple.
    // Any of the argument pointers is allowed to be NULL in which case
    // the corresponding variable is simply not filled out.
    void getChannelTriple(unsigned index, unsigned* depth,
                          int* ieta, unsigned* iphi) const;

    // The following assumes a valid HBHE depth/ieta combination
    static HcalSubdetector getSubdetector(unsigned depth, int ieta);

private:
    class HBHEChannelId
    {
    public:
        inline HBHEChannelId() : depth_(1000U), ieta_(1000), iphi_(1000U) {}

        inline HBHEChannelId(const unsigned i_depth,
                             const int i_ieta,
                             const unsigned i_iphi)
            : depth_(i_depth), ieta_(i_ieta), iphi_(i_iphi) {}

        // Inspectors
        inline unsigned depth() const {return depth_;}
        inline int ieta() const {return ieta_;}
        inline unsigned iphi() const {return iphi_;}

        inline bool operator<(const HBHEChannelId& r) const
        {
            if (depth_ < r.depth_) return true;
            if (r.depth_ < depth_) return false;
            if (ieta_ < r.ieta_) return true;
            if (r.ieta_ < ieta_) return false;
            return iphi_ < r.iphi_;
        }

        inline bool operator==(const HBHEChannelId& r) const
            {return depth_ == r.depth_ && ieta_ == r.ieta_ && iphi_ == r.iphi_;}

        inline bool operator!=(const HBHEChannelId& r) const
            {return !(*this == r);}

    private:
        unsigned depth_;
        int ieta_;
        unsigned iphi_;
    };

    typedef std::pair<HBHEChannelId,unsigned> MapPair;
    typedef std::vector<MapPair> ChannelMap;

    unsigned find(unsigned depth, int ieta, unsigned iphi) const;

    HBHEChannelId lookup_[ChannelCount];
    ChannelMap inverse_;
};

// Standard map
const HBHELinearMap& hbheChannelMap();

#endif // CondFormats_HcalObjects_HBHELinearMap_h_
