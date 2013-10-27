#ifndef CondFormats_SiStripObjects_Serialization_H
#define CondFormats_SiStripObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void FedChannelConnection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fecCrate_);
    ar & BOOST_SERIALIZATION_NVP(fecSlot_);
    ar & BOOST_SERIALIZATION_NVP(fecRing_);
    ar & BOOST_SERIALIZATION_NVP(ccuAddr_);
    ar & BOOST_SERIALIZATION_NVP(ccuChan_);
    ar & BOOST_SERIALIZATION_NVP(apv0_);
    ar & BOOST_SERIALIZATION_NVP(apv1_);
    ar & BOOST_SERIALIZATION_NVP(dcuId_);
    ar & BOOST_SERIALIZATION_NVP(detId_);
    ar & BOOST_SERIALIZATION_NVP(nApvPairs_);
    ar & BOOST_SERIALIZATION_NVP(fedCrate_);
    ar & BOOST_SERIALIZATION_NVP(fedSlot_);
    ar & BOOST_SERIALIZATION_NVP(fedId_);
    ar & BOOST_SERIALIZATION_NVP(fedCh_);
    ar & BOOST_SERIALIZATION_NVP(length_);
    ar & BOOST_SERIALIZATION_NVP(dcu0x00_);
    ar & BOOST_SERIALIZATION_NVP(mux0x43_);
    ar & BOOST_SERIALIZATION_NVP(pll0x44_);
    ar & BOOST_SERIALIZATION_NVP(lld0x60_);
}

template <class Archive>
void SiStripApvGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_gains);
    ar & BOOST_SERIALIZATION_NVP(v_detids);
    ar & BOOST_SERIALIZATION_NVP(v_ibegin);
    ar & BOOST_SERIALIZATION_NVP(v_iend);
}

template <class Archive>
void SiStripBackPlaneCorrection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_BPC);
}

template <class Archive>
void SiStripBadStrip::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_badstrips);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}

template <class Archive>
void SiStripBadStrip::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}

template <class Archive>
void SiStripBaseDelay::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(delays_);
}

template <class Archive>
void SiStripBaseDelay::Delay::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detId);
    ar & BOOST_SERIALIZATION_NVP(coarseDelay);
    ar & BOOST_SERIALIZATION_NVP(fineDelay);
}

template <class Archive>
void SiStripConfObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(parameters);
}

template <class Archive>
void SiStripDetVOff::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_Voff);
}

template <class Archive>
void SiStripFedCabling::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(feds_);
    ar & BOOST_SERIALIZATION_NVP(registry_);
    ar & BOOST_SERIALIZATION_NVP(connections_);
    ar & BOOST_SERIALIZATION_NVP(detected_);
    ar & BOOST_SERIALIZATION_NVP(undetected_);
}

template <class Archive>
void SiStripLatency::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(latencies_);
}

template <class Archive>
void SiStripLatency::Latency::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detIdAndApv);
    ar & BOOST_SERIALIZATION_NVP(latency);
    ar & BOOST_SERIALIZATION_NVP(mode);
}

template <class Archive>
void SiStripLorentzAngle::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_LA);
}

template <class Archive>
void SiStripNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}

template <class Archive>
void SiStripNoises::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}

template <class Archive>
void SiStripPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_pedestals);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}

template <class Archive>
void SiStripPedestals::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}

template <class Archive>
void SiStripRunSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(runSummary_);
}

template <class Archive>
void SiStripSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(userDBContent_);
    ar & BOOST_SERIALIZATION_NVP(v_sum_);
    ar & BOOST_SERIALIZATION_NVP(indexes_);
    ar & BOOST_SERIALIZATION_NVP(runNr_);
    ar & BOOST_SERIALIZATION_NVP(timeValue_);
}

template <class Archive>
void SiStripSummary::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
}

template <class Archive>
void SiStripThreshold::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_threshold);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}

template <class Archive>
void SiStripThreshold::Data::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(FirstStrip_and_Hth);
    ar & BOOST_SERIALIZATION_NVP(lowTh);
    ar & BOOST_SERIALIZATION_NVP(clusTh);
}

template <class Archive>
void SiStripThreshold::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}

namespace cond {
namespace serialization {

template <>
struct access<FedChannelConnection>
{
    static bool equal_(const FedChannelConnection & first, const FedChannelConnection & second)
    {
        return true
            and (equal(first.fecCrate_, second.fecCrate_))
            and (equal(first.fecSlot_, second.fecSlot_))
            and (equal(first.fecRing_, second.fecRing_))
            and (equal(first.ccuAddr_, second.ccuAddr_))
            and (equal(first.ccuChan_, second.ccuChan_))
            and (equal(first.apv0_, second.apv0_))
            and (equal(first.apv1_, second.apv1_))
            and (equal(first.dcuId_, second.dcuId_))
            and (equal(first.detId_, second.detId_))
            and (equal(first.nApvPairs_, second.nApvPairs_))
            and (equal(first.fedCrate_, second.fedCrate_))
            and (equal(first.fedSlot_, second.fedSlot_))
            and (equal(first.fedId_, second.fedId_))
            and (equal(first.fedCh_, second.fedCh_))
            and (equal(first.length_, second.length_))
            and (equal(first.dcu0x00_, second.dcu0x00_))
            and (equal(first.mux0x43_, second.mux0x43_))
            and (equal(first.pll0x44_, second.pll0x44_))
            and (equal(first.lld0x60_, second.lld0x60_))
        ;
    }
};

template <>
struct access<SiStripApvGain>
{
    static bool equal_(const SiStripApvGain & first, const SiStripApvGain & second)
    {
        return true
            and (equal(first.v_gains, second.v_gains))
            and (equal(first.v_detids, second.v_detids))
            and (equal(first.v_ibegin, second.v_ibegin))
            and (equal(first.v_iend, second.v_iend))
        ;
    }
};

template <>
struct access<SiStripBackPlaneCorrection>
{
    static bool equal_(const SiStripBackPlaneCorrection & first, const SiStripBackPlaneCorrection & second)
    {
        return true
            and (equal(first.m_BPC, second.m_BPC))
        ;
    }
};

template <>
struct access<SiStripBadStrip>
{
    static bool equal_(const SiStripBadStrip & first, const SiStripBadStrip & second)
    {
        return true
            and (equal(first.v_badstrips, second.v_badstrips))
            and (equal(first.indexes, second.indexes))
        ;
    }
};

template <>
struct access<SiStripBadStrip::DetRegistry>
{
    static bool equal_(const SiStripBadStrip::DetRegistry & first, const SiStripBadStrip::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
        ;
    }
};

template <>
struct access<SiStripBaseDelay>
{
    static bool equal_(const SiStripBaseDelay & first, const SiStripBaseDelay & second)
    {
        return true
            and (equal(first.delays_, second.delays_))
        ;
    }
};

template <>
struct access<SiStripBaseDelay::Delay>
{
    static bool equal_(const SiStripBaseDelay::Delay & first, const SiStripBaseDelay::Delay & second)
    {
        return true
            and (equal(first.detId, second.detId))
            and (equal(first.coarseDelay, second.coarseDelay))
            and (equal(first.fineDelay, second.fineDelay))
        ;
    }
};

template <>
struct access<SiStripConfObject>
{
    static bool equal_(const SiStripConfObject & first, const SiStripConfObject & second)
    {
        return true
            and (equal(first.parameters, second.parameters))
        ;
    }
};

template <>
struct access<SiStripDetVOff>
{
    static bool equal_(const SiStripDetVOff & first, const SiStripDetVOff & second)
    {
        return true
            and (equal(first.v_Voff, second.v_Voff))
        ;
    }
};

template <>
struct access<SiStripFedCabling>
{
    static bool equal_(const SiStripFedCabling & first, const SiStripFedCabling & second)
    {
        return true
            and (equal(first.feds_, second.feds_))
            and (equal(first.registry_, second.registry_))
            and (equal(first.connections_, second.connections_))
            and (equal(first.detected_, second.detected_))
            and (equal(first.undetected_, second.undetected_))
        ;
    }
};

template <>
struct access<SiStripLatency>
{
    static bool equal_(const SiStripLatency & first, const SiStripLatency & second)
    {
        return true
            and (equal(first.latencies_, second.latencies_))
        ;
    }
};

template <>
struct access<SiStripLatency::Latency>
{
    static bool equal_(const SiStripLatency::Latency & first, const SiStripLatency::Latency & second)
    {
        return true
            and (equal(first.detIdAndApv, second.detIdAndApv))
            and (equal(first.latency, second.latency))
            and (equal(first.mode, second.mode))
        ;
    }
};

template <>
struct access<SiStripLorentzAngle>
{
    static bool equal_(const SiStripLorentzAngle & first, const SiStripLorentzAngle & second)
    {
        return true
            and (equal(first.m_LA, second.m_LA))
        ;
    }
};

template <>
struct access<SiStripNoises>
{
    static bool equal_(const SiStripNoises & first, const SiStripNoises & second)
    {
        return true
            and (equal(first.v_noises, second.v_noises))
            and (equal(first.indexes, second.indexes))
        ;
    }
};

template <>
struct access<SiStripNoises::DetRegistry>
{
    static bool equal_(const SiStripNoises::DetRegistry & first, const SiStripNoises::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
        ;
    }
};

template <>
struct access<SiStripPedestals>
{
    static bool equal_(const SiStripPedestals & first, const SiStripPedestals & second)
    {
        return true
            and (equal(first.v_pedestals, second.v_pedestals))
            and (equal(first.indexes, second.indexes))
        ;
    }
};

template <>
struct access<SiStripPedestals::DetRegistry>
{
    static bool equal_(const SiStripPedestals::DetRegistry & first, const SiStripPedestals::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
        ;
    }
};

template <>
struct access<SiStripRunSummary>
{
    static bool equal_(const SiStripRunSummary & first, const SiStripRunSummary & second)
    {
        return true
            and (equal(first.runSummary_, second.runSummary_))
        ;
    }
};

template <>
struct access<SiStripSummary>
{
    static bool equal_(const SiStripSummary & first, const SiStripSummary & second)
    {
        return true
            and (equal(first.userDBContent_, second.userDBContent_))
            and (equal(first.v_sum_, second.v_sum_))
            and (equal(first.indexes_, second.indexes_))
            and (equal(first.runNr_, second.runNr_))
            and (equal(first.timeValue_, second.timeValue_))
        ;
    }
};

template <>
struct access<SiStripSummary::DetRegistry>
{
    static bool equal_(const SiStripSummary::DetRegistry & first, const SiStripSummary::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
        ;
    }
};

template <>
struct access<SiStripThreshold>
{
    static bool equal_(const SiStripThreshold & first, const SiStripThreshold & second)
    {
        return true
            and (equal(first.v_threshold, second.v_threshold))
            and (equal(first.indexes, second.indexes))
        ;
    }
};

template <>
struct access<SiStripThreshold::Data>
{
    static bool equal_(const SiStripThreshold::Data & first, const SiStripThreshold::Data & second)
    {
        return true
            and (equal(first.FirstStrip_and_Hth, second.FirstStrip_and_Hth))
            and (equal(first.lowTh, second.lowTh))
            and (equal(first.clusTh, second.clusTh))
        ;
    }
};

template <>
struct access<SiStripThreshold::DetRegistry>
{
    static bool equal_(const SiStripThreshold::DetRegistry & first, const SiStripThreshold::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
            and (equal(first.iend, second.iend))
        ;
    }
};

}
}

#endif
