
#include "CondFormats/RPCObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

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
COND_SERIALIZATION_INSTANTIATE(ChamberLocationSpec);

template <class Archive>
void ChamberStripSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cablePinNumber);
    ar & BOOST_SERIALIZATION_NVP(chamberStripNumber);
    ar & BOOST_SERIALIZATION_NVP(cmsStripNumber);
}
COND_SERIALIZATION_INSTANTIATE(ChamberStripSpec);

template <class Archive>
void DccSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theId);
    ar & BOOST_SERIALIZATION_NVP(theTBs);
}
COND_SERIALIZATION_INSTANTIATE(DccSpec);

template <class Archive>
void FebConnectorSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardInputNum);
    ar & BOOST_SERIALIZATION_NVP(theChamber);
    ar & BOOST_SERIALIZATION_NVP(theFeb);
    ar & BOOST_SERIALIZATION_NVP(theAlgo);
}
COND_SERIALIZATION_INSTANTIATE(FebConnectorSpec);

template <class Archive>
void FebLocationSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cmsEtaPartition);
    ar & BOOST_SERIALIZATION_NVP(positionInCmsEtaPartition);
    ar & BOOST_SERIALIZATION_NVP(localEtaPartition);
    ar & BOOST_SERIALIZATION_NVP(positionInLocalEtaPartition);
}
COND_SERIALIZATION_INSTANTIATE(FebLocationSpec);

template <class Archive>
void L1RPCConeBuilder::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_firstTower);
    ar & BOOST_SERIALIZATION_NVP(m_lastTower);
    ar & BOOST_SERIALIZATION_NVP(m_coneConnectionMap);
    ar & BOOST_SERIALIZATION_NVP(m_compressedConeConnectionMap);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeBuilder);

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
COND_SERIALIZATION_INSTANTIATE(L1RPCConeBuilder::TCompressedCon);

template <class Archive>
void L1RPCConeBuilder::TStripCon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_PAC);
    ar & BOOST_SERIALIZATION_NVP(m_logplane);
    ar & BOOST_SERIALIZATION_NVP(m_logstrip);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeBuilder::TStripCon);

template <class Archive>
void L1RPCDevCoords::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_tower);
    ar & BOOST_SERIALIZATION_NVP(m_PAC);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCDevCoords);

template <class Archive>
void L1RPCHwConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_disabledDevices);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCHwConfig);

template <class Archive>
void LinkBoardSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theMaster);
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardNumInLink);
    ar & BOOST_SERIALIZATION_NVP(theCode);
    ar & BOOST_SERIALIZATION_NVP(theFebs);
}
COND_SERIALIZATION_INSTANTIATE(LinkBoardSpec);

template <class Archive>
void LinkConnSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theTriggerBoardInputNumber);
    ar & BOOST_SERIALIZATION_NVP(theLBs);
}
COND_SERIALIZATION_INSTANTIATE(LinkConnSpec);

template <class Archive>
void RBCBoardSpecs::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_boardspecs);
}
COND_SERIALIZATION_INSTANTIATE(RBCBoardSpecs);

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
COND_SERIALIZATION_INSTANTIATE(RBCBoardSpecs::RBCBoardConfig);

template <class Archive>
void RPCClusterSize::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_cls);
}
COND_SERIALIZATION_INSTANTIATE(RPCClusterSize);

template <class Archive>
void RPCClusterSize::ClusterSizeItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(clusterSize);
}
COND_SERIALIZATION_INSTANTIATE(RPCClusterSize::ClusterSizeItem);

template <class Archive>
void RPCDQMObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dqmv);
    ar & BOOST_SERIALIZATION_NVP(run);
    ar & BOOST_SERIALIZATION_NVP(v_cls);
}
COND_SERIALIZATION_INSTANTIATE(RPCDQMObject);

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
COND_SERIALIZATION_INSTANTIATE(RPCDQMObject::DQMObjectItem);

template <class Archive>
void RPCDeadStrips::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(DeadVec);
}
COND_SERIALIZATION_INSTANTIATE(RPCDeadStrips);

template <class Archive>
void RPCDeadStrips::DeadItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(rawId);
    ar & BOOST_SERIALIZATION_NVP(strip);
}
COND_SERIALIZATION_INSTANTIATE(RPCDeadStrips::DeadItem);

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
COND_SERIALIZATION_INSTANTIATE(RPCEMap);

template <class Archive>
void RPCEMap::dccItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theId);
    ar & BOOST_SERIALIZATION_NVP(nTBs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::dccItem);

template <class Archive>
void RPCEMap::febItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardInputNum);
    ar & BOOST_SERIALIZATION_NVP(thePartition);
    ar & BOOST_SERIALIZATION_NVP(theChamber);
    ar & BOOST_SERIALIZATION_NVP(theAlgo);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::febItem);

template <class Archive>
void RPCEMap::lbItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theMaster);
    ar & BOOST_SERIALIZATION_NVP(theLinkBoardNumInLink);
    ar & BOOST_SERIALIZATION_NVP(theCode);
    ar & BOOST_SERIALIZATION_NVP(nFebs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::lbItem);

template <class Archive>
void RPCEMap::linkItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theTriggerBoardInputNumber);
    ar & BOOST_SERIALIZATION_NVP(nLBs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::linkItem);

template <class Archive>
void RPCEMap::tbItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theNum);
    ar & BOOST_SERIALIZATION_NVP(nLinks);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::tbItem);

template <class Archive>
void RPCMaskedStrips::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(MaskVec);
}
COND_SERIALIZATION_INSTANTIATE(RPCMaskedStrips);

template <class Archive>
void RPCMaskedStrips::MaskItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(rawId);
    ar & BOOST_SERIALIZATION_NVP(strip);
}
COND_SERIALIZATION_INSTANTIATE(RPCMaskedStrips::MaskItem);

template <class Archive>
void RPCObFebmap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObFebMap_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObFebmap);

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
COND_SERIALIZATION_INSTANTIATE(RPCObFebmap::Feb_Item);

template <class Archive>
void RPCObGas::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGas_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGas);

template <class Archive>
void RPCObGas::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(flowin);
    ar & BOOST_SERIALIZATION_NVP(flowout);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGas::Item);

template <class Archive>
void RPCObGasHum::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGasHum_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasHum);

template <class Archive>
void RPCObGasHum::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(unixtime);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(dpid);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasHum::Item);

template <class Archive>
void RPCObGasMix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGasMix_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasMix);

template <class Archive>
void RPCObGasMix::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(unixtime);
    ar & BOOST_SERIALIZATION_NVP(gas1);
    ar & BOOST_SERIALIZATION_NVP(gas2);
    ar & BOOST_SERIALIZATION_NVP(gas3);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasMix::Item);

template <class Archive>
void RPCObGasmap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObGasMap_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasmap);

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
COND_SERIALIZATION_INSTANTIATE(RPCObGasmap::GasMap_Item);

template <class Archive>
void RPCObImon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObImon_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObImon);

template <class Archive>
void RPCObImon::I_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObImon::I_Item);

template <class Archive>
void RPCObPVSSmap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObIDMap_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObPVSSmap);

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
COND_SERIALIZATION_INSTANTIATE(RPCObPVSSmap::Item);

template <class Archive>
void RPCObStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObStatus_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObStatus);

template <class Archive>
void RPCObStatus::S_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObStatus::S_Item);

template <class Archive>
void RPCObTemp::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObTemp_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObTemp);

template <class Archive>
void RPCObTemp::T_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObTemp::T_Item);

template <class Archive>
void RPCObUXC::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObUXC_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObUXC);

template <class Archive>
void RPCObUXC::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(temperature);
    ar & BOOST_SERIALIZATION_NVP(pressure);
    ar & BOOST_SERIALIZATION_NVP(dewpoint);
    ar & BOOST_SERIALIZATION_NVP(unixtime);
}
COND_SERIALIZATION_INSTANTIATE(RPCObUXC::Item);

template <class Archive>
void RPCObVmon::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ObVmon_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObVmon);

template <class Archive>
void RPCObVmon::V_Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(value);
    ar & BOOST_SERIALIZATION_NVP(day);
    ar & BOOST_SERIALIZATION_NVP(time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObVmon::V_Item);

template <class Archive>
void RPCReadOutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theFeds);
    ar & BOOST_SERIALIZATION_NVP(theVersion);
}
COND_SERIALIZATION_INSTANTIATE(RPCReadOutMapping);

template <class Archive>
void RPCStripNoises::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(v_noises);
    ar & BOOST_SERIALIZATION_NVP(v_cls);
}
COND_SERIALIZATION_INSTANTIATE(RPCStripNoises);

template <class Archive>
void RPCStripNoises::NoiseItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(dpid);
    ar & BOOST_SERIALIZATION_NVP(noise);
    ar & BOOST_SERIALIZATION_NVP(eff);
    ar & BOOST_SERIALIZATION_NVP(time);
}
COND_SERIALIZATION_INSTANTIATE(RPCStripNoises::NoiseItem);

template <class Archive>
void RPCTechTriggerConfig::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_runId);
    ar & BOOST_SERIALIZATION_NVP(m_runType);
    ar & BOOST_SERIALIZATION_NVP(m_triggerMode);
}
COND_SERIALIZATION_INSTANTIATE(RPCTechTriggerConfig);

template <class Archive>
void TTUBoardSpecs::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_boardspecs);
}
COND_SERIALIZATION_INSTANTIATE(TTUBoardSpecs);

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
COND_SERIALIZATION_INSTANTIATE(TTUBoardSpecs::TTUBoardConfig);

template <class Archive>
void TriggerBoardSpec::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theNum);
    ar & BOOST_SERIALIZATION_NVP(theMaskedLinks);
    ar & BOOST_SERIALIZATION_NVP(theLinks);
}
COND_SERIALIZATION_INSTANTIATE(TriggerBoardSpec);

#include "CondFormats/RPCObjects/src/SerializationManual.h"
