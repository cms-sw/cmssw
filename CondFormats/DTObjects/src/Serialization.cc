
#include "CondFormats/DTObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void DTCCBConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(timeStamp);
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(fullConfigKey);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTCCBConfig);

template <class Archive>
void DTCCBId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
}
COND_SERIALIZATION_INSTANTIATE(DTCCBId);

template <class Archive>
void DTConfigKey::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(confType);
    ar & BOOST_SERIALIZATION_NVP(confKey);
}
COND_SERIALIZATION_INSTANTIATE(DTConfigKey);

template <class Archive>
void DTDeadFlag::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTDeadFlag);

template <class Archive>
void DTDeadFlagData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dead_HV);
    ar & BOOST_SERIALIZATION_NVP(dead_TP);
    ar & BOOST_SERIALIZATION_NVP(dead_RO);
    ar & BOOST_SERIALIZATION_NVP(discCat);
}
COND_SERIALIZATION_INSTANTIATE(DTDeadFlagData);

template <class Archive>
void DTDeadFlagId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
    ar & BOOST_SERIALIZATION_NVP(layerId);
    ar & BOOST_SERIALIZATION_NVP(cellId);
}
COND_SERIALIZATION_INSTANTIATE(DTDeadFlagId);

template <class Archive>
void DTHVStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTHVStatus);

template <class Archive>
void DTHVStatusData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fCell);
    ar & BOOST_SERIALIZATION_NVP(lCell);
    ar & BOOST_SERIALIZATION_NVP(flagA);
    ar & BOOST_SERIALIZATION_NVP(flagC);
    ar & BOOST_SERIALIZATION_NVP(flagS);
}
COND_SERIALIZATION_INSTANTIATE(DTHVStatusData);

template <class Archive>
void DTHVStatusId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
    ar & BOOST_SERIALIZATION_NVP(layerId);
    ar & BOOST_SERIALIZATION_NVP(partId);
}
COND_SERIALIZATION_INSTANTIATE(DTHVStatusId);

template <class Archive>
void DTKeyedConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::BaseKeyed", boost::serialization::base_object<cond::BaseKeyed>(*this));
    ar & BOOST_SERIALIZATION_NVP(cfgId);
    ar & BOOST_SERIALIZATION_NVP(dataList);
    ar & BOOST_SERIALIZATION_NVP(linkList);
}
COND_SERIALIZATION_INSTANTIATE(DTKeyedConfig);

template <class Archive>
void DTLVStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTLVStatus);

template <class Archive>
void DTLVStatusData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(flagCFE);
    ar & BOOST_SERIALIZATION_NVP(flagDFE);
    ar & BOOST_SERIALIZATION_NVP(flagCMC);
    ar & BOOST_SERIALIZATION_NVP(flagDMC);
}
COND_SERIALIZATION_INSTANTIATE(DTLVStatusData);

template <class Archive>
void DTLVStatusId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
}
COND_SERIALIZATION_INSTANTIATE(DTLVStatusId);

template <class Archive>
void DTMtime::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTMtime);

template <class Archive>
void DTMtimeData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mTime);
    ar & BOOST_SERIALIZATION_NVP(mTrms);
}
COND_SERIALIZATION_INSTANTIATE(DTMtimeData);

template <class Archive>
void DTMtimeId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
    ar & BOOST_SERIALIZATION_NVP(layerId);
    ar & BOOST_SERIALIZATION_NVP(cellId);
}
COND_SERIALIZATION_INSTANTIATE(DTMtimeId);

template <class Archive>
void DTPerformance::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTPerformance);

template <class Archive>
void DTPerformanceData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(meanT0);
    ar & BOOST_SERIALIZATION_NVP(meanTtrig);
    ar & BOOST_SERIALIZATION_NVP(meanMtime);
    ar & BOOST_SERIALIZATION_NVP(meanNoise);
    ar & BOOST_SERIALIZATION_NVP(meanAfterPulse);
    ar & BOOST_SERIALIZATION_NVP(meanResolution);
    ar & BOOST_SERIALIZATION_NVP(meanEfficiency);
}
COND_SERIALIZATION_INSTANTIATE(DTPerformanceData);

template <class Archive>
void DTPerformanceId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
}
COND_SERIALIZATION_INSTANTIATE(DTPerformanceId);

template <class Archive>
void DTRangeT0::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTRangeT0);

template <class Archive>
void DTRangeT0Data::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t0min);
    ar & BOOST_SERIALIZATION_NVP(t0max);
}
COND_SERIALIZATION_INSTANTIATE(DTRangeT0Data);

template <class Archive>
void DTRangeT0Id::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
}
COND_SERIALIZATION_INSTANTIATE(DTRangeT0Id);

template <class Archive>
void DTReadOutGeometryLink::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dduId);
    ar & BOOST_SERIALIZATION_NVP(rosId);
    ar & BOOST_SERIALIZATION_NVP(robId);
    ar & BOOST_SERIALIZATION_NVP(tdcId);
    ar & BOOST_SERIALIZATION_NVP(channelId);
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
    ar & BOOST_SERIALIZATION_NVP(layerId);
    ar & BOOST_SERIALIZATION_NVP(cellId);
}
COND_SERIALIZATION_INSTANTIATE(DTReadOutGeometryLink);

template <class Archive>
void DTReadOutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cellMapVersion);
    ar & BOOST_SERIALIZATION_NVP(robMapVersion);
    ar & BOOST_SERIALIZATION_NVP(readOutChannelDriftTubeMap);
}
COND_SERIALIZATION_INSTANTIATE(DTReadOutMapping);

template <class Archive>
void DTStatusFlag::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTStatusFlag);

template <class Archive>
void DTStatusFlagData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(noiseFlag);
    ar & BOOST_SERIALIZATION_NVP(feMask);
    ar & BOOST_SERIALIZATION_NVP(tdcMask);
    ar & BOOST_SERIALIZATION_NVP(trigMask);
    ar & BOOST_SERIALIZATION_NVP(deadFlag);
    ar & BOOST_SERIALIZATION_NVP(nohvFlag);
}
COND_SERIALIZATION_INSTANTIATE(DTStatusFlagData);

template <class Archive>
void DTStatusFlagId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
    ar & BOOST_SERIALIZATION_NVP(layerId);
    ar & BOOST_SERIALIZATION_NVP(cellId);
}
COND_SERIALIZATION_INSTANTIATE(DTStatusFlagId);

template <class Archive>
void DTT0::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTT0);

template <class Archive>
void DTT0Data::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(channelId);
    ar & BOOST_SERIALIZATION_NVP(t0mean);
    ar & BOOST_SERIALIZATION_NVP(t0rms);
}
COND_SERIALIZATION_INSTANTIATE(DTT0Data);

template <class Archive>
void DTTPGParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(clockLength);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTTPGParameters);

template <class Archive>
void DTTPGParametersData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(nClock);
    ar & BOOST_SERIALIZATION_NVP(tPhase);
}
COND_SERIALIZATION_INSTANTIATE(DTTPGParametersData);

template <class Archive>
void DTTPGParametersId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
}
COND_SERIALIZATION_INSTANTIATE(DTTPGParametersId);

template <class Archive>
void DTTtrig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}
COND_SERIALIZATION_INSTANTIATE(DTTtrig);

template <class Archive>
void DTTtrigData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(tTrig);
    ar & BOOST_SERIALIZATION_NVP(tTrms);
    ar & BOOST_SERIALIZATION_NVP(kFact);
}
COND_SERIALIZATION_INSTANTIATE(DTTtrigData);

template <class Archive>
void DTTtrigId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
    ar & BOOST_SERIALIZATION_NVP(layerId);
    ar & BOOST_SERIALIZATION_NVP(cellId);
}
COND_SERIALIZATION_INSTANTIATE(DTTtrigId);

#include "CondFormats/DTObjects/src/SerializationManual.h"
