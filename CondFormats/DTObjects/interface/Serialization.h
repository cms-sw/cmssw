#ifndef CondFormats_DTObjects_Serialization_H
#define CondFormats_DTObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void DTCCBConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(timeStamp);
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(fullConfigKey);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTCCBId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
}

template <class Archive>
void DTConfigKey::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(confType);
    ar & BOOST_SERIALIZATION_NVP(confKey);
}

template <class Archive>
void DTDeadFlag::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTDeadFlagData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dead_HV);
    ar & BOOST_SERIALIZATION_NVP(dead_TP);
    ar & BOOST_SERIALIZATION_NVP(dead_RO);
    ar & BOOST_SERIALIZATION_NVP(discCat);
}

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

template <class Archive>
void DTHVStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTHVStatusData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fCell);
    ar & BOOST_SERIALIZATION_NVP(lCell);
    ar & BOOST_SERIALIZATION_NVP(flagA);
    ar & BOOST_SERIALIZATION_NVP(flagC);
    ar & BOOST_SERIALIZATION_NVP(flagS);
}

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

template <class Archive>
void DTKeyedConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::BaseKeyed", boost::serialization::base_object<cond::BaseKeyed>(*this));
    ar & BOOST_SERIALIZATION_NVP(cfgId);
    ar & BOOST_SERIALIZATION_NVP(dataList);
    ar & BOOST_SERIALIZATION_NVP(linkList);
}

template <class Archive>
void DTLVStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTLVStatusData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(flagCFE);
    ar & BOOST_SERIALIZATION_NVP(flagDFE);
    ar & BOOST_SERIALIZATION_NVP(flagCMC);
    ar & BOOST_SERIALIZATION_NVP(flagDMC);
}

template <class Archive>
void DTLVStatusId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
}

template <class Archive>
void DTMtime::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTMtimeData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mTime);
    ar & BOOST_SERIALIZATION_NVP(mTrms);
}

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

template <class Archive>
void DTPerformance::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

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

template <class Archive>
void DTPerformanceId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
}

template <class Archive>
void DTRangeT0::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTRangeT0Data::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t0min);
    ar & BOOST_SERIALIZATION_NVP(t0max);
}

template <class Archive>
void DTRangeT0Id::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
    ar & BOOST_SERIALIZATION_NVP(slId);
}

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

template <class Archive>
void DTReadOutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cellMapVersion);
    ar & BOOST_SERIALIZATION_NVP(robMapVersion);
    ar & BOOST_SERIALIZATION_NVP(readOutChannelDriftTubeMap);
}

template <class Archive>
void DTStatusFlag::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

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

template <class Archive>
void DTT0::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTT0Data::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(channelId);
    ar & BOOST_SERIALIZATION_NVP(t0mean);
    ar & BOOST_SERIALIZATION_NVP(t0rms);
}

template <class Archive>
void DTTPGParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(clockLength);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTTPGParametersData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(nClock);
    ar & BOOST_SERIALIZATION_NVP(tPhase);
}

template <class Archive>
void DTTPGParametersId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wheelId);
    ar & BOOST_SERIALIZATION_NVP(stationId);
    ar & BOOST_SERIALIZATION_NVP(sectorId);
}

template <class Archive>
void DTTtrig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dataVersion);
    ar & BOOST_SERIALIZATION_NVP(nsPerCount);
    ar & BOOST_SERIALIZATION_NVP(dataList);
}

template <class Archive>
void DTTtrigData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(tTrig);
    ar & BOOST_SERIALIZATION_NVP(tTrms);
    ar & BOOST_SERIALIZATION_NVP(kFact);
}

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

namespace cond {
namespace serialization {

template <>
struct access<DTCCBConfig>
{
    static bool equal_(const DTCCBConfig & first, const DTCCBConfig & second)
    {
        return true
            and (equal(first.timeStamp, second.timeStamp))
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.fullConfigKey, second.fullConfigKey))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTCCBId>
{
    static bool equal_(const DTCCBId & first, const DTCCBId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
        ;
    }
};

template <>
struct access<DTConfigKey>
{
    static bool equal_(const DTConfigKey & first, const DTConfigKey & second)
    {
        return true
            and (equal(first.confType, second.confType))
            and (equal(first.confKey, second.confKey))
        ;
    }
};

template <>
struct access<DTDeadFlag>
{
    static bool equal_(const DTDeadFlag & first, const DTDeadFlag & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTDeadFlagData>
{
    static bool equal_(const DTDeadFlagData & first, const DTDeadFlagData & second)
    {
        return true
            and (equal(first.dead_HV, second.dead_HV))
            and (equal(first.dead_TP, second.dead_TP))
            and (equal(first.dead_RO, second.dead_RO))
            and (equal(first.discCat, second.discCat))
        ;
    }
};

template <>
struct access<DTDeadFlagId>
{
    static bool equal_(const DTDeadFlagId & first, const DTDeadFlagId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
            and (equal(first.layerId, second.layerId))
            and (equal(first.cellId, second.cellId))
        ;
    }
};

template <>
struct access<DTHVStatus>
{
    static bool equal_(const DTHVStatus & first, const DTHVStatus & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTHVStatusData>
{
    static bool equal_(const DTHVStatusData & first, const DTHVStatusData & second)
    {
        return true
            and (equal(first.fCell, second.fCell))
            and (equal(first.lCell, second.lCell))
            and (equal(first.flagA, second.flagA))
            and (equal(first.flagC, second.flagC))
            and (equal(first.flagS, second.flagS))
        ;
    }
};

template <>
struct access<DTHVStatusId>
{
    static bool equal_(const DTHVStatusId & first, const DTHVStatusId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
            and (equal(first.layerId, second.layerId))
            and (equal(first.partId, second.partId))
        ;
    }
};

template <>
struct access<DTKeyedConfig>
{
    static bool equal_(const DTKeyedConfig & first, const DTKeyedConfig & second)
    {
        return true
            and (equal(static_cast<const cond::BaseKeyed &>(first), static_cast<const cond::BaseKeyed &>(second)))
            and (equal(first.cfgId, second.cfgId))
            and (equal(first.dataList, second.dataList))
            and (equal(first.linkList, second.linkList))
        ;
    }
};

template <>
struct access<DTLVStatus>
{
    static bool equal_(const DTLVStatus & first, const DTLVStatus & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTLVStatusData>
{
    static bool equal_(const DTLVStatusData & first, const DTLVStatusData & second)
    {
        return true
            and (equal(first.flagCFE, second.flagCFE))
            and (equal(first.flagDFE, second.flagDFE))
            and (equal(first.flagCMC, second.flagCMC))
            and (equal(first.flagDMC, second.flagDMC))
        ;
    }
};

template <>
struct access<DTLVStatusId>
{
    static bool equal_(const DTLVStatusId & first, const DTLVStatusId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
        ;
    }
};

template <>
struct access<DTMtime>
{
    static bool equal_(const DTMtime & first, const DTMtime & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.nsPerCount, second.nsPerCount))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTMtimeData>
{
    static bool equal_(const DTMtimeData & first, const DTMtimeData & second)
    {
        return true
            and (equal(first.mTime, second.mTime))
            and (equal(first.mTrms, second.mTrms))
        ;
    }
};

template <>
struct access<DTMtimeId>
{
    static bool equal_(const DTMtimeId & first, const DTMtimeId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
            and (equal(first.layerId, second.layerId))
            and (equal(first.cellId, second.cellId))
        ;
    }
};

template <>
struct access<DTPerformance>
{
    static bool equal_(const DTPerformance & first, const DTPerformance & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.nsPerCount, second.nsPerCount))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTPerformanceData>
{
    static bool equal_(const DTPerformanceData & first, const DTPerformanceData & second)
    {
        return true
            and (equal(first.meanT0, second.meanT0))
            and (equal(first.meanTtrig, second.meanTtrig))
            and (equal(first.meanMtime, second.meanMtime))
            and (equal(first.meanNoise, second.meanNoise))
            and (equal(first.meanAfterPulse, second.meanAfterPulse))
            and (equal(first.meanResolution, second.meanResolution))
            and (equal(first.meanEfficiency, second.meanEfficiency))
        ;
    }
};

template <>
struct access<DTPerformanceId>
{
    static bool equal_(const DTPerformanceId & first, const DTPerformanceId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
        ;
    }
};

template <>
struct access<DTRangeT0>
{
    static bool equal_(const DTRangeT0 & first, const DTRangeT0 & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTRangeT0Data>
{
    static bool equal_(const DTRangeT0Data & first, const DTRangeT0Data & second)
    {
        return true
            and (equal(first.t0min, second.t0min))
            and (equal(first.t0max, second.t0max))
        ;
    }
};

template <>
struct access<DTRangeT0Id>
{
    static bool equal_(const DTRangeT0Id & first, const DTRangeT0Id & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
        ;
    }
};

template <>
struct access<DTReadOutGeometryLink>
{
    static bool equal_(const DTReadOutGeometryLink & first, const DTReadOutGeometryLink & second)
    {
        return true
            and (equal(first.dduId, second.dduId))
            and (equal(first.rosId, second.rosId))
            and (equal(first.robId, second.robId))
            and (equal(first.tdcId, second.tdcId))
            and (equal(first.channelId, second.channelId))
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
            and (equal(first.layerId, second.layerId))
            and (equal(first.cellId, second.cellId))
        ;
    }
};

template <>
struct access<DTReadOutMapping>
{
    static bool equal_(const DTReadOutMapping & first, const DTReadOutMapping & second)
    {
        return true
            and (equal(first.cellMapVersion, second.cellMapVersion))
            and (equal(first.robMapVersion, second.robMapVersion))
            and (equal(first.readOutChannelDriftTubeMap, second.readOutChannelDriftTubeMap))
        ;
    }
};

template <>
struct access<DTStatusFlag>
{
    static bool equal_(const DTStatusFlag & first, const DTStatusFlag & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTStatusFlagData>
{
    static bool equal_(const DTStatusFlagData & first, const DTStatusFlagData & second)
    {
        return true
            and (equal(first.noiseFlag, second.noiseFlag))
            and (equal(first.feMask, second.feMask))
            and (equal(first.tdcMask, second.tdcMask))
            and (equal(first.trigMask, second.trigMask))
            and (equal(first.deadFlag, second.deadFlag))
            and (equal(first.nohvFlag, second.nohvFlag))
        ;
    }
};

template <>
struct access<DTStatusFlagId>
{
    static bool equal_(const DTStatusFlagId & first, const DTStatusFlagId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
            and (equal(first.layerId, second.layerId))
            and (equal(first.cellId, second.cellId))
        ;
    }
};

template <>
struct access<DTT0>
{
    static bool equal_(const DTT0 & first, const DTT0 & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.nsPerCount, second.nsPerCount))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTT0Data>
{
    static bool equal_(const DTT0Data & first, const DTT0Data & second)
    {
        return true
            and (equal(first.channelId, second.channelId))
            and (equal(first.t0mean, second.t0mean))
            and (equal(first.t0rms, second.t0rms))
        ;
    }
};

template <>
struct access<DTTPGParameters>
{
    static bool equal_(const DTTPGParameters & first, const DTTPGParameters & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.nsPerCount, second.nsPerCount))
            and (equal(first.clockLength, second.clockLength))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTTPGParametersData>
{
    static bool equal_(const DTTPGParametersData & first, const DTTPGParametersData & second)
    {
        return true
            and (equal(first.nClock, second.nClock))
            and (equal(first.tPhase, second.tPhase))
        ;
    }
};

template <>
struct access<DTTPGParametersId>
{
    static bool equal_(const DTTPGParametersId & first, const DTTPGParametersId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
        ;
    }
};

template <>
struct access<DTTtrig>
{
    static bool equal_(const DTTtrig & first, const DTTtrig & second)
    {
        return true
            and (equal(first.dataVersion, second.dataVersion))
            and (equal(first.nsPerCount, second.nsPerCount))
            and (equal(first.dataList, second.dataList))
        ;
    }
};

template <>
struct access<DTTtrigData>
{
    static bool equal_(const DTTtrigData & first, const DTTtrigData & second)
    {
        return true
            and (equal(first.tTrig, second.tTrig))
            and (equal(first.tTrms, second.tTrms))
            and (equal(first.kFact, second.kFact))
        ;
    }
};

template <>
struct access<DTTtrigId>
{
    static bool equal_(const DTTtrigId & first, const DTTtrigId & second)
    {
        return true
            and (equal(first.wheelId, second.wheelId))
            and (equal(first.stationId, second.stationId))
            and (equal(first.sectorId, second.sectorId))
            and (equal(first.slId, second.slId))
            and (equal(first.layerId, second.layerId))
            and (equal(first.cellId, second.cellId))
        ;
    }
};

}
}

#endif
