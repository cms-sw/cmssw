#ifndef CondFormats_RPCObjects_Serialization_H
#define CondFormats_RPCObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void ChamberLocationSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(diskOrWheel);
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(sector);
    ar & BOOST_SERIALIZATION_NVP(subsector);
    ar & BOOST_SERIALIZATION_NVP(febZOrnt);
    ar & BOOST_SERIALIZATION_NVP(febZRadOrnt);
    ar & BOOST_SERIALIZATION_NVP(barrelOrEndcap);
}

template <class Archive>
void ChamberStripSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cablePinNumber);
    ar & BOOST_SERIALIZATION_NVP(chamberStripNumber);
    ar & BOOST_SERIALIZATION_NVP(cmsStripNumber);
}

template <class Archive>
void DccSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theId);
    ar & BOOST_SERIALIZATION_NVP(theTBs);
}

template <class Archive>
void FebConnectorSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardInputNum);
    ar & BOOST_SERIALIZATION_NVP(theChamber);
    ar & BOOST_SERIALIZATION_NVP(theFeb);
    ar & BOOST_SERIALIZATION_NVP(theAlgo);
}

template <class Archive>
void FebLocationSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cmsEtaPartition);
    ar & BOOST_SERIALIZATION_NVP(positionInCmsEtaPartition);
    ar & BOOST_SERIALIZATION_NVP(localEtaPartition);
    ar & BOOST_SERIALIZATION_NVP(positionInLocalEtaPartition);
}

template <class Archive>
void L1RPCConeBuilder::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_firstTower);
    ar & BOOST_SERIALIZATION_NVP(m_lastTower);
    ar & BOOST_SERIALIZATION_NVP(m_coneConnectionMap);
    ar & BOOST_SERIALIZATION_NVP(m_compressedConeConnectionMap);
}

template <class Archive>
void L1RPCConeBuilder::TCompressedCon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_mul);
    ar & BOOST_SERIALIZATION_NVP(m_PAC);
    ar & BOOST_SERIALIZATION_NVP(m_logplane);
    ar & BOOST_SERIALIZATION_NVP(m_validForStripFirst);
    ar & BOOST_SERIALIZATION_NVP(m_validForStripLast);
    ar & BOOST_SERIALIZATION_NVP(m_offset);
}

template <class Archive>
void L1RPCConeBuilder::TStripCon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_PAC);
    ar & BOOST_SERIALIZATION_NVP(m_logplane);
    ar & BOOST_SERIALIZATION_NVP(m_logstrip);
}

template <class Archive>
void L1RPCDevCoords::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_PAC);
}

template <class Archive>
void L1RPCHwConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_disabledDevices);
}

template <class Archive>
void LinkBoardSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theMaster);
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardNumInLink);
    ar & BOOST_SERIALIZATION_NVP(theCode);
    ar & BOOST_SERIALIZATION_NVP(theFebs);
}

template <class Archive>
void LinkConnSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theTriggerBoardInputNumber);
    ar & BOOST_SERIALIZATION_NVP(theLBs);
}

template <class Archive>
void RBCBoardSpecs::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_boardspecs);
}

template <class Archive>
void RBCBoardSpecs::RBCBoardConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_Firmware);
    ar & BOOST_SERIALIZATION_NVP(m_WheelId);
    ar & BOOST_SERIALIZATION_NVP(m_Latency);
    ar & BOOST_SERIALIZATION_NVP(m_MayorityLevel);
    ar & BOOST_SERIALIZATION_NVP(m_MaskedOrInput);
    ar & BOOST_SERIALIZATION_NVP(m_ForcedOrInput);
    ar & BOOST_SERIALIZATION_NVP(m_LogicType);
}

template <class Archive>
void RPCClusterSize::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_cls);
}

template <class Archive>
void RPCClusterSize::ClusterSizeItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(clusterSize);
}

template <class Archive>
void RPCDQMObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dqmv);
    ar & BOOST_SERIALIZATION_NVP(run);
    ar & BOOST_SERIALIZATION_NVP(v_cls);
}

template <class Archive>
void RPCDQMObject::DQMObjectItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(clusterSize);
    ar & BOOST_SERIALIZATION_NVP(bx);
    ar & BOOST_SERIALIZATION_NVP(bxrms);
    ar & BOOST_SERIALIZATION_NVP(efficiency);
    ar & BOOST_SERIALIZATION_NVP(numdigi);
    ar & BOOST_SERIALIZATION_NVP(numcluster);
    ar & BOOST_SERIALIZATION_NVP(status);
    ar & BOOST_SERIALIZATION_NVP(weight);
}

template <class Archive>
void RPCDeadStrips::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(DeadVec);
}

template <class Archive>
void RPCDeadStrips::DeadItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(rawId);
    ar & BOOST_SERIALIZATION_NVP(strip);
}

template <class Archive>
void RPCEMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theVersion);
    ar & BOOST_SERIALIZATION_NVP(theDccs);
    ar & BOOST_SERIALIZATION_NVP(theTBs);
    ar & BOOST_SERIALIZATION_NVP(theLinks);
    ar & BOOST_SERIALIZATION_NVP(theLBs);
    ar & BOOST_SERIALIZATION_NVP(theFebs);
}

template <class Archive>
void RPCEMap::dccItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theId);
    ar & BOOST_SERIALIZATION_NVP(nTBs);
}

template <class Archive>
void RPCEMap::febItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardInputNum);
    ar & BOOST_SERIALIZATION_NVP(thePartition);
    ar & BOOST_SERIALIZATION_NVP(theChamber);
    ar & BOOST_SERIALIZATION_NVP(theAlgo);
}

template <class Archive>
void RPCEMap::lbItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theMaster);
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardNumInLink);
    ar & BOOST_SERIALIZATION_NVP(theCode);
    ar & BOOST_SERIALIZATION_NVP(nFebs);
}

template <class Archive>
void RPCEMap::linkItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theTriggerBoardInputNumber);
    ar & BOOST_SERIALIZATION_NVP(nLBs);
}

template <class Archive>
void RPCEMap::tbItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theNum);
    ar & BOOST_SERIALIZATION_NVP(nLinks);
}

template <class Archive>
void RPCMaskedStrips::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(MaskVec);
}

template <class Archive>
void RPCMaskedStrips::MaskItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(rawId);
    ar & BOOST_SERIALIZATION_NVP(strip);
}

template <class Archive>
void RPCObFebmap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObFebMap_rpc);
}

template <class Archive>
void RPCObFebmap::Feb_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(thr1);
    ar & BOOST_SERIALIZATION_NVP(thr2);
    ar & BOOST_SERIALIZATION_NVP(thr3);
    ar & BOOST_SERIALIZATION_NVP(thr4);
    ar & BOOST_SERIALIZATION_NVP(vmon1);
    ar & BOOST_SERIALIZATION_NVP(vmon2);
    ar & BOOST_SERIALIZATION_NVP(vmon3);
    ar & BOOST_SERIALIZATION_NVP(vmon4);
    ar & BOOST_SERIALIZATION_NVP(temp1);
    ar & BOOST_SERIALIZATION_NVP(temp2);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
    ar & BOOST_SERIALIZATION_NVP(noise1);
    ar & BOOST_SERIALIZATION_NVP(noise2);
    ar & BOOST_SERIALIZATION_NVP(noise3);
    ar & BOOST_SERIALIZATION_NVP(noise4);
}

template <class Archive>
void RPCObGas::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGas_rpc);
}

template <class Archive>
void RPCObGas::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(flowin);
    ar & BOOST_SERIALIZATION_NVP(flowout);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}

template <class Archive>
void RPCObGasHum::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGasHum_rpc);
}

template <class Archive>
void RPCObGasHum::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(unixtime);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(dpid);
}

template <class Archive>
void RPCObGasMix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGasMix_rpc);
}

template <class Archive>
void RPCObGasMix::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(unixtime);
    ar & BOOST_SERIALIZATION_NVP(gas1);
    ar & BOOST_SERIALIZATION_NVP(gas2);
    ar & BOOST_SERIALIZATION_NVP(gas3);
}

template <class Archive>
void RPCObGasmap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGasMap_rpc);
}

template <class Archive>
void RPCObGasmap::GasMap_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(region);
    ar & BOOST_SERIALIZATION_NVP(ring);
    ar & BOOST_SERIALIZATION_NVP(station);
    ar & BOOST_SERIALIZATION_NVP(sector);
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(subsector);
    ar & BOOST_SERIALIZATION_NVP(suptype);
}

template <class Archive>
void RPCObImon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObImon_rpc);
}

template <class Archive>
void RPCObImon::I_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}

template <class Archive>
void RPCObPVSSmap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObIDMap_rpc);
}

template <class Archive>
void RPCObPVSSmap::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(since);
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(region);
    ar & BOOST_SERIALIZATION_NVP(ring);
    ar & BOOST_SERIALIZATION_NVP(station);
    ar & BOOST_SERIALIZATION_NVP(sector);
    ar & BOOST_SERIALIZATION_NVP(layer);
    ar & BOOST_SERIALIZATION_NVP(subsector);
    ar & BOOST_SERIALIZATION_NVP(suptype);
}

template <class Archive>
void RPCObStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObStatus_rpc);
}

template <class Archive>
void RPCObStatus::S_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}

template <class Archive>
void RPCObTemp::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObTemp_rpc);
}

template <class Archive>
void RPCObTemp::T_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}

template <class Archive>
void RPCObUXC::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObUXC_rpc);
}

template <class Archive>
void RPCObUXC::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(temperature);
    ar & BOOST_SERIALIZATION_NVP(pressure);
    ar & BOOST_SERIALIZATION_NVP(dewpoint);
    ar & BOOST_SERIALIZATION_NVP(unixtime);
}

template <class Archive>
void RPCObVmon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObVmon_rpc);
}

template <class Archive>
void RPCObVmon::V_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}

template <class Archive>
void RPCReadOutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theFeds);
    ar & BOOST_SERIALIZATION_NVP(theVersion);
}

template <class Archive>
void RPCStripNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(v_cls);
}

template <class Archive>
void RPCStripNoises::NoiseItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(noise);
    ar & BOOST_SERIALIZATION_NVP(eff);
    ar & BOOST_SERIALIZATION_NVP(time);
}

template <class Archive>
void RPCTechTriggerConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_runId);
    ar & BOOST_SERIALIZATION_NVP(m_runType);
    ar & BOOST_SERIALIZATION_NVP(m_triggerMode);
}

template <class Archive>
void TTUBoardSpecs::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_boardspecs);
}

template <class Archive>
void TTUBoardSpecs::TTUBoardConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("RPCTechTriggerConfig", boost::serialization::base_object<RPCTechTriggerConfig>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_Firmware);
    ar & BOOST_SERIALIZATION_NVP(m_LengthOfFiber);
    ar & BOOST_SERIALIZATION_NVP(m_Delay);
    ar & BOOST_SERIALIZATION_NVP(m_MaxNumWheels);
    ar & BOOST_SERIALIZATION_NVP(m_Wheel1Id);
    ar & BOOST_SERIALIZATION_NVP(m_Wheel2Id);
    ar & BOOST_SERIALIZATION_NVP(m_TrackLength);
    ar & BOOST_SERIALIZATION_NVP(m_MaskedSectors);
    ar & BOOST_SERIALIZATION_NVP(m_ForcedSectors);
    ar & BOOST_SERIALIZATION_NVP(m_LogicType);
}

template <class Archive>
void TriggerBoardSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theNum);
    ar & BOOST_SERIALIZATION_NVP(theMaskedLinks);
    ar & BOOST_SERIALIZATION_NVP(theLinks);
}

namespace cond {
namespace serialization {

template <>
struct access<ChamberLocationSpec>
{
    static bool equal_(const ChamberLocationSpec & first, const ChamberLocationSpec & second)
    {
        return true
            and (equal(first.diskOrWheel, second.diskOrWheel))
            and (equal(first.layer, second.layer))
            and (equal(first.sector, second.sector))
            and (equal(first.subsector, second.subsector))
            and (equal(first.febZOrnt, second.febZOrnt))
            and (equal(first.febZRadOrnt, second.febZRadOrnt))
            and (equal(first.barrelOrEndcap, second.barrelOrEndcap))
        ;
    }
};

template <>
struct access<ChamberStripSpec>
{
    static bool equal_(const ChamberStripSpec & first, const ChamberStripSpec & second)
    {
        return true
            and (equal(first.cablePinNumber, second.cablePinNumber))
            and (equal(first.chamberStripNumber, second.chamberStripNumber))
            and (equal(first.cmsStripNumber, second.cmsStripNumber))
        ;
    }
};

template <>
struct access<DccSpec>
{
    static bool equal_(const DccSpec & first, const DccSpec & second)
    {
        return true
            and (equal(first.theId, second.theId))
            and (equal(first.theTBs, second.theTBs))
        ;
    }
};

template <>
struct access<FebConnectorSpec>
{
    static bool equal_(const FebConnectorSpec & first, const FebConnectorSpec & second)
    {
        return true
            and (equal(first.theLinkBoardInputNum, second.theLinkBoardInputNum))
            and (equal(first.theChamber, second.theChamber))
            and (equal(first.theFeb, second.theFeb))
            and (equal(first.theAlgo, second.theAlgo))
        ;
    }
};

template <>
struct access<FebLocationSpec>
{
    static bool equal_(const FebLocationSpec & first, const FebLocationSpec & second)
    {
        return true
            and (equal(first.cmsEtaPartition, second.cmsEtaPartition))
            and (equal(first.positionInCmsEtaPartition, second.positionInCmsEtaPartition))
            and (equal(first.localEtaPartition, second.localEtaPartition))
            and (equal(first.positionInLocalEtaPartition, second.positionInLocalEtaPartition))
        ;
    }
};

template <>
struct access<L1RPCConeBuilder>
{
    static bool equal_(const L1RPCConeBuilder & first, const L1RPCConeBuilder & second)
    {
        return true
            and (equal(first.m_firstTower, second.m_firstTower))
            and (equal(first.m_lastTower, second.m_lastTower))
            and (equal(first.m_coneConnectionMap, second.m_coneConnectionMap))
            and (equal(first.m_compressedConeConnectionMap, second.m_compressedConeConnectionMap))
        ;
    }
};

template <>
struct access<L1RPCConeBuilder::TCompressedCon>
{
    static bool equal_(const L1RPCConeBuilder::TCompressedCon & first, const L1RPCConeBuilder::TCompressedCon & second)
    {
        return true
            and (equal(first.m_tower, second.m_tower))
            and (equal(first.m_mul, second.m_mul))
            and (equal(first.m_PAC, second.m_PAC))
            and (equal(first.m_logplane, second.m_logplane))
            and (equal(first.m_validForStripFirst, second.m_validForStripFirst))
            and (equal(first.m_validForStripLast, second.m_validForStripLast))
            and (equal(first.m_offset, second.m_offset))
        ;
    }
};

template <>
struct access<L1RPCConeBuilder::TStripCon>
{
    static bool equal_(const L1RPCConeBuilder::TStripCon & first, const L1RPCConeBuilder::TStripCon & second)
    {
        return true
            and (equal(first.m_tower, second.m_tower))
            and (equal(first.m_PAC, second.m_PAC))
            and (equal(first.m_logplane, second.m_logplane))
            and (equal(first.m_logstrip, second.m_logstrip))
        ;
    }
};

template <>
struct access<L1RPCDevCoords>
{
    static bool equal_(const L1RPCDevCoords & first, const L1RPCDevCoords & second)
    {
        return true
            and (equal(first.m_tower, second.m_tower))
            and (equal(first.m_PAC, second.m_PAC))
        ;
    }
};

template <>
struct access<L1RPCHwConfig>
{
    static bool equal_(const L1RPCHwConfig & first, const L1RPCHwConfig & second)
    {
        return true
            and (equal(first.m_disabledDevices, second.m_disabledDevices))
        ;
    }
};

template <>
struct access<LinkBoardSpec>
{
    static bool equal_(const LinkBoardSpec & first, const LinkBoardSpec & second)
    {
        return true
            and (equal(first.theMaster, second.theMaster))
            and (equal(first.theLinkBoardNumInLink, second.theLinkBoardNumInLink))
            and (equal(first.theCode, second.theCode))
            and (equal(first.theFebs, second.theFebs))
        ;
    }
};

template <>
struct access<LinkConnSpec>
{
    static bool equal_(const LinkConnSpec & first, const LinkConnSpec & second)
    {
        return true
            and (equal(first.theTriggerBoardInputNumber, second.theTriggerBoardInputNumber))
            and (equal(first.theLBs, second.theLBs))
        ;
    }
};

template <>
struct access<RBCBoardSpecs>
{
    static bool equal_(const RBCBoardSpecs & first, const RBCBoardSpecs & second)
    {
        return true
            and (equal(first.v_boardspecs, second.v_boardspecs))
        ;
    }
};

template <>
struct access<RBCBoardSpecs::RBCBoardConfig>
{
    static bool equal_(const RBCBoardSpecs::RBCBoardConfig & first, const RBCBoardSpecs::RBCBoardConfig & second)
    {
        return true
            and (equal(first.m_Firmware, second.m_Firmware))
            and (equal(first.m_WheelId, second.m_WheelId))
            and (equal(first.m_Latency, second.m_Latency))
            and (equal(first.m_MayorityLevel, second.m_MayorityLevel))
            and (equal(first.m_MaskedOrInput, second.m_MaskedOrInput))
            and (equal(first.m_ForcedOrInput, second.m_ForcedOrInput))
            and (equal(first.m_LogicType, second.m_LogicType))
        ;
    }
};

template <>
struct access<RPCClusterSize>
{
    static bool equal_(const RPCClusterSize & first, const RPCClusterSize & second)
    {
        return true
            and (equal(first.v_cls, second.v_cls))
        ;
    }
};

template <>
struct access<RPCClusterSize::ClusterSizeItem>
{
    static bool equal_(const RPCClusterSize::ClusterSizeItem & first, const RPCClusterSize::ClusterSizeItem & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.clusterSize, second.clusterSize))
        ;
    }
};

template <>
struct access<RPCDQMObject>
{
    static bool equal_(const RPCDQMObject & first, const RPCDQMObject & second)
    {
        return true
            and (equal(first.dqmv, second.dqmv))
            and (equal(first.run, second.run))
            and (equal(first.v_cls, second.v_cls))
        ;
    }
};

template <>
struct access<RPCDQMObject::DQMObjectItem>
{
    static bool equal_(const RPCDQMObject::DQMObjectItem & first, const RPCDQMObject::DQMObjectItem & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.clusterSize, second.clusterSize))
            and (equal(first.bx, second.bx))
            and (equal(first.bxrms, second.bxrms))
            and (equal(first.efficiency, second.efficiency))
            and (equal(first.numdigi, second.numdigi))
            and (equal(first.numcluster, second.numcluster))
            and (equal(first.status, second.status))
            and (equal(first.weight, second.weight))
        ;
    }
};

template <>
struct access<RPCDeadStrips>
{
    static bool equal_(const RPCDeadStrips & first, const RPCDeadStrips & second)
    {
        return true
            and (equal(first.DeadVec, second.DeadVec))
        ;
    }
};

template <>
struct access<RPCDeadStrips::DeadItem>
{
    static bool equal_(const RPCDeadStrips::DeadItem & first, const RPCDeadStrips::DeadItem & second)
    {
        return true
            and (equal(first.rawId, second.rawId))
            and (equal(first.strip, second.strip))
        ;
    }
};

template <>
struct access<RPCEMap>
{
    static bool equal_(const RPCEMap & first, const RPCEMap & second)
    {
        return true
            and (equal(first.theVersion, second.theVersion))
            and (equal(first.theDccs, second.theDccs))
            and (equal(first.theTBs, second.theTBs))
            and (equal(first.theLinks, second.theLinks))
            and (equal(first.theLBs, second.theLBs))
            and (equal(first.theFebs, second.theFebs))
        ;
    }
};

template <>
struct access<RPCEMap::dccItem>
{
    static bool equal_(const RPCEMap::dccItem & first, const RPCEMap::dccItem & second)
    {
        return true
            and (equal(first.theId, second.theId))
            and (equal(first.nTBs, second.nTBs))
        ;
    }
};

template <>
struct access<RPCEMap::febItem>
{
    static bool equal_(const RPCEMap::febItem & first, const RPCEMap::febItem & second)
    {
        return true
            and (equal(first.theLinkBoardInputNum, second.theLinkBoardInputNum))
            and (equal(first.thePartition, second.thePartition))
            and (equal(first.theChamber, second.theChamber))
            and (equal(first.theAlgo, second.theAlgo))
        ;
    }
};

template <>
struct access<RPCEMap::lbItem>
{
    static bool equal_(const RPCEMap::lbItem & first, const RPCEMap::lbItem & second)
    {
        return true
            and (equal(first.theMaster, second.theMaster))
            and (equal(first.theLinkBoardNumInLink, second.theLinkBoardNumInLink))
            and (equal(first.theCode, second.theCode))
            and (equal(first.nFebs, second.nFebs))
        ;
    }
};

template <>
struct access<RPCEMap::linkItem>
{
    static bool equal_(const RPCEMap::linkItem & first, const RPCEMap::linkItem & second)
    {
        return true
            and (equal(first.theTriggerBoardInputNumber, second.theTriggerBoardInputNumber))
            and (equal(first.nLBs, second.nLBs))
        ;
    }
};

template <>
struct access<RPCEMap::tbItem>
{
    static bool equal_(const RPCEMap::tbItem & first, const RPCEMap::tbItem & second)
    {
        return true
            and (equal(first.theNum, second.theNum))
            and (equal(first.nLinks, second.nLinks))
        ;
    }
};

template <>
struct access<RPCMaskedStrips>
{
    static bool equal_(const RPCMaskedStrips & first, const RPCMaskedStrips & second)
    {
        return true
            and (equal(first.MaskVec, second.MaskVec))
        ;
    }
};

template <>
struct access<RPCMaskedStrips::MaskItem>
{
    static bool equal_(const RPCMaskedStrips::MaskItem & first, const RPCMaskedStrips::MaskItem & second)
    {
        return true
            and (equal(first.rawId, second.rawId))
            and (equal(first.strip, second.strip))
        ;
    }
};

template <>
struct access<RPCObFebmap>
{
    static bool equal_(const RPCObFebmap & first, const RPCObFebmap & second)
    {
        return true
            and (equal(first.ObFebMap_rpc, second.ObFebMap_rpc))
        ;
    }
};

template <>
struct access<RPCObFebmap::Feb_Item>
{
    static bool equal_(const RPCObFebmap::Feb_Item & first, const RPCObFebmap::Feb_Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.thr1, second.thr1))
            and (equal(first.thr2, second.thr2))
            and (equal(first.thr3, second.thr3))
            and (equal(first.thr4, second.thr4))
            and (equal(first.vmon1, second.vmon1))
            and (equal(first.vmon2, second.vmon2))
            and (equal(first.vmon3, second.vmon3))
            and (equal(first.vmon4, second.vmon4))
            and (equal(first.temp1, second.temp1))
            and (equal(first.temp2, second.temp2))
            and (equal(first.day, second.day))
            and (equal(first.time, second.time))
            and (equal(first.noise1, second.noise1))
            and (equal(first.noise2, second.noise2))
            and (equal(first.noise3, second.noise3))
            and (equal(first.noise4, second.noise4))
        ;
    }
};

template <>
struct access<RPCObGas>
{
    static bool equal_(const RPCObGas & first, const RPCObGas & second)
    {
        return true
            and (equal(first.ObGas_rpc, second.ObGas_rpc))
        ;
    }
};

template <>
struct access<RPCObGas::Item>
{
    static bool equal_(const RPCObGas::Item & first, const RPCObGas::Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.flowin, second.flowin))
            and (equal(first.flowout, second.flowout))
            and (equal(first.day, second.day))
            and (equal(first.time, second.time))
        ;
    }
};

template <>
struct access<RPCObGasHum>
{
    static bool equal_(const RPCObGasHum & first, const RPCObGasHum & second)
    {
        return true
            and (equal(first.ObGasHum_rpc, second.ObGasHum_rpc))
        ;
    }
};

template <>
struct access<RPCObGasHum::Item>
{
    static bool equal_(const RPCObGasHum::Item & first, const RPCObGasHum::Item & second)
    {
        return true
            and (equal(first.unixtime, second.unixtime))
            and (equal(first.value, second.value))
            and (equal(first.dpid, second.dpid))
        ;
    }
};

template <>
struct access<RPCObGasMix>
{
    static bool equal_(const RPCObGasMix & first, const RPCObGasMix & second)
    {
        return true
            and (equal(first.ObGasMix_rpc, second.ObGasMix_rpc))
        ;
    }
};

template <>
struct access<RPCObGasMix::Item>
{
    static bool equal_(const RPCObGasMix::Item & first, const RPCObGasMix::Item & second)
    {
        return true
            and (equal(first.unixtime, second.unixtime))
            and (equal(first.gas1, second.gas1))
            and (equal(first.gas2, second.gas2))
            and (equal(first.gas3, second.gas3))
        ;
    }
};

template <>
struct access<RPCObGasmap>
{
    static bool equal_(const RPCObGasmap & first, const RPCObGasmap & second)
    {
        return true
            and (equal(first.ObGasMap_rpc, second.ObGasMap_rpc))
        ;
    }
};

template <>
struct access<RPCObGasmap::GasMap_Item>
{
    static bool equal_(const RPCObGasmap::GasMap_Item & first, const RPCObGasmap::GasMap_Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.region, second.region))
            and (equal(first.ring, second.ring))
            and (equal(first.station, second.station))
            and (equal(first.sector, second.sector))
            and (equal(first.layer, second.layer))
            and (equal(first.subsector, second.subsector))
            and (equal(first.suptype, second.suptype))
        ;
    }
};

template <>
struct access<RPCObImon>
{
    static bool equal_(const RPCObImon & first, const RPCObImon & second)
    {
        return true
            and (equal(first.ObImon_rpc, second.ObImon_rpc))
        ;
    }
};

template <>
struct access<RPCObImon::I_Item>
{
    static bool equal_(const RPCObImon::I_Item & first, const RPCObImon::I_Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.value, second.value))
            and (equal(first.day, second.day))
            and (equal(first.time, second.time))
        ;
    }
};

template <>
struct access<RPCObPVSSmap>
{
    static bool equal_(const RPCObPVSSmap & first, const RPCObPVSSmap & second)
    {
        return true
            and (equal(first.ObIDMap_rpc, second.ObIDMap_rpc))
        ;
    }
};

template <>
struct access<RPCObPVSSmap::Item>
{
    static bool equal_(const RPCObPVSSmap::Item & first, const RPCObPVSSmap::Item & second)
    {
        return true
            and (equal(first.since, second.since))
            and (equal(first.dpid, second.dpid))
            and (equal(first.region, second.region))
            and (equal(first.ring, second.ring))
            and (equal(first.station, second.station))
            and (equal(first.sector, second.sector))
            and (equal(first.layer, second.layer))
            and (equal(first.subsector, second.subsector))
            and (equal(first.suptype, second.suptype))
        ;
    }
};

template <>
struct access<RPCObStatus>
{
    static bool equal_(const RPCObStatus & first, const RPCObStatus & second)
    {
        return true
            and (equal(first.ObStatus_rpc, second.ObStatus_rpc))
        ;
    }
};

template <>
struct access<RPCObStatus::S_Item>
{
    static bool equal_(const RPCObStatus::S_Item & first, const RPCObStatus::S_Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.value, second.value))
            and (equal(first.day, second.day))
            and (equal(first.time, second.time))
        ;
    }
};

template <>
struct access<RPCObTemp>
{
    static bool equal_(const RPCObTemp & first, const RPCObTemp & second)
    {
        return true
            and (equal(first.ObTemp_rpc, second.ObTemp_rpc))
        ;
    }
};

template <>
struct access<RPCObTemp::T_Item>
{
    static bool equal_(const RPCObTemp::T_Item & first, const RPCObTemp::T_Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.value, second.value))
            and (equal(first.day, second.day))
            and (equal(first.time, second.time))
        ;
    }
};

template <>
struct access<RPCObUXC>
{
    static bool equal_(const RPCObUXC & first, const RPCObUXC & second)
    {
        return true
            and (equal(first.ObUXC_rpc, second.ObUXC_rpc))
        ;
    }
};

template <>
struct access<RPCObUXC::Item>
{
    static bool equal_(const RPCObUXC::Item & first, const RPCObUXC::Item & second)
    {
        return true
            and (equal(first.temperature, second.temperature))
            and (equal(first.pressure, second.pressure))
            and (equal(first.dewpoint, second.dewpoint))
            and (equal(first.unixtime, second.unixtime))
        ;
    }
};

template <>
struct access<RPCObVmon>
{
    static bool equal_(const RPCObVmon & first, const RPCObVmon & second)
    {
        return true
            and (equal(first.ObVmon_rpc, second.ObVmon_rpc))
        ;
    }
};

template <>
struct access<RPCObVmon::V_Item>
{
    static bool equal_(const RPCObVmon::V_Item & first, const RPCObVmon::V_Item & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.value, second.value))
            and (equal(first.day, second.day))
            and (equal(first.time, second.time))
        ;
    }
};

template <>
struct access<RPCReadOutMapping>
{
    static bool equal_(const RPCReadOutMapping & first, const RPCReadOutMapping & second)
    {
        return true
            and (equal(first.theFeds, second.theFeds))
            and (equal(first.theVersion, second.theVersion))
        ;
    }
};

template <>
struct access<RPCStripNoises>
{
    static bool equal_(const RPCStripNoises & first, const RPCStripNoises & second)
    {
        return true
            and (equal(first.v_noises, second.v_noises))
            and (equal(first.v_cls, second.v_cls))
        ;
    }
};

template <>
struct access<RPCStripNoises::NoiseItem>
{
    static bool equal_(const RPCStripNoises::NoiseItem & first, const RPCStripNoises::NoiseItem & second)
    {
        return true
            and (equal(first.dpid, second.dpid))
            and (equal(first.noise, second.noise))
            and (equal(first.eff, second.eff))
            and (equal(first.time, second.time))
        ;
    }
};

template <>
struct access<RPCTechTriggerConfig>
{
    static bool equal_(const RPCTechTriggerConfig & first, const RPCTechTriggerConfig & second)
    {
        return true
            and (equal(first.m_runId, second.m_runId))
            and (equal(first.m_runType, second.m_runType))
            and (equal(first.m_triggerMode, second.m_triggerMode))
        ;
    }
};

template <>
struct access<TTUBoardSpecs>
{
    static bool equal_(const TTUBoardSpecs & first, const TTUBoardSpecs & second)
    {
        return true
            and (equal(first.m_boardspecs, second.m_boardspecs))
        ;
    }
};

template <>
struct access<TTUBoardSpecs::TTUBoardConfig>
{
    static bool equal_(const TTUBoardSpecs::TTUBoardConfig & first, const TTUBoardSpecs::TTUBoardConfig & second)
    {
        return true
            and (equal(static_cast<const RPCTechTriggerConfig &>(first), static_cast<const RPCTechTriggerConfig &>(second)))
            and (equal(first.m_Firmware, second.m_Firmware))
            and (equal(first.m_LengthOfFiber, second.m_LengthOfFiber))
            and (equal(first.m_Delay, second.m_Delay))
            and (equal(first.m_MaxNumWheels, second.m_MaxNumWheels))
            and (equal(first.m_Wheel1Id, second.m_Wheel1Id))
            and (equal(first.m_Wheel2Id, second.m_Wheel2Id))
            and (equal(first.m_TrackLength, second.m_TrackLength))
            and (equal(first.m_MaskedSectors, second.m_MaskedSectors))
            and (equal(first.m_ForcedSectors, second.m_ForcedSectors))
            and (equal(first.m_LogicType, second.m_LogicType))
        ;
    }
};

template <>
struct access<TriggerBoardSpec>
{
    static bool equal_(const TriggerBoardSpec & first, const TriggerBoardSpec & second)
    {
        return true
            and (equal(first.theNum, second.theNum))
            and (equal(first.theMaskedLinks, second.theMaskedLinks))
            and (equal(first.theLinks, second.theLinks))
        ;
    }
};

}
}

#endif
