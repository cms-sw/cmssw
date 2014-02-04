
#include "CondFormats/SiStripObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

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
COND_SERIALIZATION_INSTANTIATE(FedChannelConnection);

template <class Archive>
void SiStripApvGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_gains);
    ar & BOOST_SERIALIZATION_NVP(v_detids);
    ar & BOOST_SERIALIZATION_NVP(v_ibegin);
    ar & BOOST_SERIALIZATION_NVP(v_iend);
}
COND_SERIALIZATION_INSTANTIATE(SiStripApvGain);

template <class Archive>
void SiStripBackPlaneCorrection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_BPC);
}
COND_SERIALIZATION_INSTANTIATE(SiStripBackPlaneCorrection);

template <class Archive>
void SiStripBadStrip::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_badstrips);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}
COND_SERIALIZATION_INSTANTIATE(SiStripBadStrip);

template <class Archive>
void SiStripBadStrip::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}
COND_SERIALIZATION_INSTANTIATE(SiStripBadStrip::DetRegistry);

template <class Archive>
void SiStripBaseDelay::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(delays_);
}
COND_SERIALIZATION_INSTANTIATE(SiStripBaseDelay);

template <class Archive>
void SiStripBaseDelay::Delay::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detId);
    ar & BOOST_SERIALIZATION_NVP(coarseDelay);
    ar & BOOST_SERIALIZATION_NVP(fineDelay);
}
COND_SERIALIZATION_INSTANTIATE(SiStripBaseDelay::Delay);

template <class Archive>
void SiStripConfObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(parameters);
}
COND_SERIALIZATION_INSTANTIATE(SiStripConfObject);

template <class Archive>
void SiStripDetVOff::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_Voff);
}
COND_SERIALIZATION_INSTANTIATE(SiStripDetVOff);

template <class Archive>
void SiStripFedCabling::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(feds_);
    ar & BOOST_SERIALIZATION_NVP(registry_);
    ar & BOOST_SERIALIZATION_NVP(connections_);
    ar & BOOST_SERIALIZATION_NVP(detected_);
    ar & BOOST_SERIALIZATION_NVP(undetected_);
}
COND_SERIALIZATION_INSTANTIATE(SiStripFedCabling);

template <class Archive>
void SiStripLatency::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(latencies_);
}
COND_SERIALIZATION_INSTANTIATE(SiStripLatency);

template <class Archive>
void SiStripLatency::Latency::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detIdAndApv);
    ar & BOOST_SERIALIZATION_NVP(latency);
    ar & BOOST_SERIALIZATION_NVP(mode);
}
COND_SERIALIZATION_INSTANTIATE(SiStripLatency::Latency);

template <class Archive>
void SiStripLorentzAngle::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_LA);
}
COND_SERIALIZATION_INSTANTIATE(SiStripLorentzAngle);

template <class Archive>
void SiStripNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}
COND_SERIALIZATION_INSTANTIATE(SiStripNoises);

template <class Archive>
void SiStripNoises::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}
COND_SERIALIZATION_INSTANTIATE(SiStripNoises::DetRegistry);

template <class Archive>
void SiStripPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_pedestals);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}
COND_SERIALIZATION_INSTANTIATE(SiStripPedestals);

template <class Archive>
void SiStripPedestals::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}
COND_SERIALIZATION_INSTANTIATE(SiStripPedestals::DetRegistry);

template <class Archive>
void SiStripRunSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(runSummary_);
}
COND_SERIALIZATION_INSTANTIATE(SiStripRunSummary);

template <class Archive>
void SiStripSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(userDBContent_);
    ar & BOOST_SERIALIZATION_NVP(v_sum_);
    ar & BOOST_SERIALIZATION_NVP(indexes_);
    ar & BOOST_SERIALIZATION_NVP(runNr_);
    ar & BOOST_SERIALIZATION_NVP(timeValue_);
}
COND_SERIALIZATION_INSTANTIATE(SiStripSummary);

template <class Archive>
void SiStripSummary::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
}
COND_SERIALIZATION_INSTANTIATE(SiStripSummary::DetRegistry);

template <class Archive>
void SiStripThreshold::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_threshold);
    ar & BOOST_SERIALIZATION_NVP(indexes);
}
COND_SERIALIZATION_INSTANTIATE(SiStripThreshold);

template <class Archive>
void SiStripThreshold::Data::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(FirstStrip_and_Hth);
    ar & BOOST_SERIALIZATION_NVP(lowTh);
    ar & BOOST_SERIALIZATION_NVP(clusTh);
}
COND_SERIALIZATION_INSTANTIATE(SiStripThreshold::Data);

template <class Archive>
void SiStripThreshold::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
    ar & BOOST_SERIALIZATION_NVP(iend);
}
COND_SERIALIZATION_INSTANTIATE(SiStripThreshold::DetRegistry);

#include "CondFormats/SiStripObjects/src/SerializationManual.h"
