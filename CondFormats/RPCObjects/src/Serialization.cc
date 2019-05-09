
#include "CondFormats/RPCObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void ChamberLocationSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("diskOrWheel", diskOrWheel);
    ar & boost::serialization::make_nvp("layer", layer);
    ar & boost::serialization::make_nvp("sector", sector);
    ar & boost::serialization::make_nvp("subsector", subsector);
    ar & boost::serialization::make_nvp("febZOrnt", febZOrnt);
    ar & boost::serialization::make_nvp("febZRadOrnt", febZRadOrnt);
    ar & boost::serialization::make_nvp("barrelOrEndcap", barrelOrEndcap);
}
COND_SERIALIZATION_INSTANTIATE(ChamberLocationSpec);

template <class Archive>
void ChamberStripSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cablePinNumber", cablePinNumber);
    ar & boost::serialization::make_nvp("chamberStripNumber", chamberStripNumber);
    ar & boost::serialization::make_nvp("cmsStripNumber", cmsStripNumber);
}
COND_SERIALIZATION_INSTANTIATE(ChamberStripSpec);

template <class Archive>
void DccSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theId", theId);
    ar & boost::serialization::make_nvp("theTBs", theTBs);
}
COND_SERIALIZATION_INSTANTIATE(DccSpec);

template <class Archive>
void FebConnectorSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theLinkBoardInputNum", theLinkBoardInputNum);
    ar & boost::serialization::make_nvp("theChamber", theChamber);
    ar & boost::serialization::make_nvp("theFeb", theFeb);
    ar & boost::serialization::make_nvp("theAlgo", theAlgo);
}
COND_SERIALIZATION_INSTANTIATE(FebConnectorSpec);

template <class Archive>
void FebLocationSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cmsEtaPartition", cmsEtaPartition);
    ar & boost::serialization::make_nvp("positionInCmsEtaPartition", positionInCmsEtaPartition);
    ar & boost::serialization::make_nvp("localEtaPartition", localEtaPartition);
    ar & boost::serialization::make_nvp("positionInLocalEtaPartition", positionInLocalEtaPartition);
}
COND_SERIALIZATION_INSTANTIATE(FebLocationSpec);

template <class Archive>
void L1RPCConeBuilder::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-firstTower", m_firstTower);
    ar & boost::serialization::make_nvp("m-lastTower", m_lastTower);
    ar & boost::serialization::make_nvp("m-coneConnectionMap", m_coneConnectionMap);
    ar & boost::serialization::make_nvp("m-compressedConeConnectionMap", m_compressedConeConnectionMap);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeBuilder);

template <class Archive>
void L1RPCConeBuilder::TCompressedCon::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-tower", m_tower);
    ar & boost::serialization::make_nvp("m-mul", m_mul);
    ar & boost::serialization::make_nvp("m-PAC", m_PAC);
    ar & boost::serialization::make_nvp("m-logplane", m_logplane);
    ar & boost::serialization::make_nvp("m-validForStripFirst", m_validForStripFirst);
    ar & boost::serialization::make_nvp("m-validForStripLast", m_validForStripLast);
    ar & boost::serialization::make_nvp("m-offset", m_offset);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeBuilder::TCompressedCon);

template <class Archive>
void L1RPCConeBuilder::TStripCon::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-tower", m_tower);
    ar & boost::serialization::make_nvp("m-PAC", m_PAC);
    ar & boost::serialization::make_nvp("m-logplane", m_logplane);
    ar & boost::serialization::make_nvp("m-logstrip", m_logstrip);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCConeBuilder::TStripCon);

template <class Archive>
void L1RPCDevCoords::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-tower", m_tower);
    ar & boost::serialization::make_nvp("m-PAC", m_PAC);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCDevCoords);

template <class Archive>
void L1RPCHwConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-disabledDevices", m_disabledDevices);
}
COND_SERIALIZATION_INSTANTIATE(L1RPCHwConfig);

template <class Archive>
void LinkBoardSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theMaster", theMaster);
    ar & boost::serialization::make_nvp("theLinkBoardNumInLink", theLinkBoardNumInLink);
    ar & boost::serialization::make_nvp("theCode", theCode);
    ar & boost::serialization::make_nvp("theFebs", theFebs);
}
COND_SERIALIZATION_INSTANTIATE(LinkBoardSpec);

template <class Archive>
void LinkConnSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theTriggerBoardInputNumber", theTriggerBoardInputNumber);
    ar & boost::serialization::make_nvp("theLBs", theLBs);
}
COND_SERIALIZATION_INSTANTIATE(LinkConnSpec);

template <class Archive>
void RBCBoardSpecs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("v-boardspecs", v_boardspecs);
}
COND_SERIALIZATION_INSTANTIATE(RBCBoardSpecs);

template <class Archive>
void RBCBoardSpecs::RBCBoardConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-Firmware", m_Firmware);
    ar & boost::serialization::make_nvp("m-WheelId", m_WheelId);
    ar & boost::serialization::make_nvp("m-Latency", m_Latency);
    ar & boost::serialization::make_nvp("m-MayorityLevel", m_MayorityLevel);
    ar & boost::serialization::make_nvp("m-MaskedOrInput", m_MaskedOrInput);
    ar & boost::serialization::make_nvp("m-ForcedOrInput", m_ForcedOrInput);
    ar & boost::serialization::make_nvp("m-LogicType", m_LogicType);
}
COND_SERIALIZATION_INSTANTIATE(RBCBoardSpecs::RBCBoardConfig);

template <class Archive>
void RPCAMCLink::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("id-", id_);
}
COND_SERIALIZATION_INSTANTIATE(RPCAMCLink);

template <class Archive>
void RPCAMCLinkMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("map-", map_);
}
COND_SERIALIZATION_INSTANTIATE(RPCAMCLinkMap);

template <class Archive>
void RPCClusterSize::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("v-cls", v_cls);
}
COND_SERIALIZATION_INSTANTIATE(RPCClusterSize);

template <class Archive>
void RPCClusterSize::ClusterSizeItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("clusterSize", clusterSize);
}
COND_SERIALIZATION_INSTANTIATE(RPCClusterSize::ClusterSizeItem);

template <class Archive>
void RPCDCCLink::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("id-", id_);
}
COND_SERIALIZATION_INSTANTIATE(RPCDCCLink);

template <class Archive>
void RPCDCCLinkMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("map-", map_);
}
COND_SERIALIZATION_INSTANTIATE(RPCDCCLinkMap);

template <class Archive>
void RPCDQMObject::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dqmv", dqmv);
    ar & boost::serialization::make_nvp("run", run);
    ar & boost::serialization::make_nvp("v-cls", v_cls);
}
COND_SERIALIZATION_INSTANTIATE(RPCDQMObject);

template <class Archive>
void RPCDQMObject::DQMObjectItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("clusterSize", clusterSize);
    ar & boost::serialization::make_nvp("bx", bx);
    ar & boost::serialization::make_nvp("bxrms", bxrms);
    ar & boost::serialization::make_nvp("efficiency", efficiency);
    ar & boost::serialization::make_nvp("numdigi", numdigi);
    ar & boost::serialization::make_nvp("numcluster", numcluster);
    ar & boost::serialization::make_nvp("status", status);
    ar & boost::serialization::make_nvp("weight", weight);
}
COND_SERIALIZATION_INSTANTIATE(RPCDQMObject::DQMObjectItem);

template <class Archive>
void RPCDeadStrips::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("DeadVec", DeadVec);
}
COND_SERIALIZATION_INSTANTIATE(RPCDeadStrips);

template <class Archive>
void RPCDeadStrips::DeadItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("rawId", rawId);
    ar & boost::serialization::make_nvp("strip", strip);
}
COND_SERIALIZATION_INSTANTIATE(RPCDeadStrips::DeadItem);

template <class Archive>
void RPCEMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theVersion", theVersion);
    ar & boost::serialization::make_nvp("theDccs", theDccs);
    ar & boost::serialization::make_nvp("theTBs", theTBs);
    ar & boost::serialization::make_nvp("theLinks", theLinks);
    ar & boost::serialization::make_nvp("theLBs", theLBs);
    ar & boost::serialization::make_nvp("theFebs", theFebs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap);

template <class Archive>
void RPCEMap::dccItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theId", theId);
    ar & boost::serialization::make_nvp("nTBs", nTBs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::dccItem);

template <class Archive>
void RPCEMap::febItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theLinkBoardInputNum", theLinkBoardInputNum);
    ar & boost::serialization::make_nvp("thePartition", thePartition);
    ar & boost::serialization::make_nvp("theChamber", theChamber);
    ar & boost::serialization::make_nvp("theAlgo", theAlgo);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::febItem);

template <class Archive>
void RPCEMap::lbItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theMaster", theMaster);
    ar & boost::serialization::make_nvp("theLinkBoardNumInLink", theLinkBoardNumInLink);
    ar & boost::serialization::make_nvp("theCode", theCode);
    ar & boost::serialization::make_nvp("nFebs", nFebs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::lbItem);

template <class Archive>
void RPCEMap::linkItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theTriggerBoardInputNumber", theTriggerBoardInputNumber);
    ar & boost::serialization::make_nvp("nLBs", nLBs);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::linkItem);

template <class Archive>
void RPCEMap::tbItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theNum", theNum);
    ar & boost::serialization::make_nvp("nLinks", nLinks);
}
COND_SERIALIZATION_INSTANTIATE(RPCEMap::tbItem);

template <class Archive>
void RPCFebConnector::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("first-strip-", first_strip_);
    ar & boost::serialization::make_nvp("slope-", slope_);
    ar & boost::serialization::make_nvp("channels-", channels_);
    ar & boost::serialization::make_nvp("rpc-det-id-", rpc_det_id_);
}
COND_SERIALIZATION_INSTANTIATE(RPCFebConnector);

template <class Archive>
void RPCLBLink::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("id-", id_);
}
COND_SERIALIZATION_INSTANTIATE(RPCLBLink);

template <class Archive>
void RPCLBLinkMap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("map-", map_);
}
COND_SERIALIZATION_INSTANTIATE(RPCLBLinkMap);

template <class Archive>
void RPCMaskedStrips::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("MaskVec", MaskVec);
}
COND_SERIALIZATION_INSTANTIATE(RPCMaskedStrips);

template <class Archive>
void RPCMaskedStrips::MaskItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("rawId", rawId);
    ar & boost::serialization::make_nvp("strip", strip);
}
COND_SERIALIZATION_INSTANTIATE(RPCMaskedStrips::MaskItem);

template <class Archive>
void RPCObFebmap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObFebMap-rpc", ObFebMap_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObFebmap);

template <class Archive>
void RPCObFebmap::Feb_Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("thr1", thr1);
    ar & boost::serialization::make_nvp("thr2", thr2);
    ar & boost::serialization::make_nvp("thr3", thr3);
    ar & boost::serialization::make_nvp("thr4", thr4);
    ar & boost::serialization::make_nvp("vmon1", vmon1);
    ar & boost::serialization::make_nvp("vmon2", vmon2);
    ar & boost::serialization::make_nvp("vmon3", vmon3);
    ar & boost::serialization::make_nvp("vmon4", vmon4);
    ar & boost::serialization::make_nvp("temp1", temp1);
    ar & boost::serialization::make_nvp("temp2", temp2);
    ar & boost::serialization::make_nvp("day", day);
    ar & boost::serialization::make_nvp("time", time);
    ar & boost::serialization::make_nvp("noise1", noise1);
    ar & boost::serialization::make_nvp("noise2", noise2);
    ar & boost::serialization::make_nvp("noise3", noise3);
    ar & boost::serialization::make_nvp("noise4", noise4);
}
COND_SERIALIZATION_INSTANTIATE(RPCObFebmap::Feb_Item);

template <class Archive>
void RPCObGas::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObGas-rpc", ObGas_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGas);

template <class Archive>
void RPCObGas::Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("flowin", flowin);
    ar & boost::serialization::make_nvp("flowout", flowout);
    ar & boost::serialization::make_nvp("day", day);
    ar & boost::serialization::make_nvp("time", time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGas::Item);

template <class Archive>
void RPCObGasHum::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObGasHum-rpc", ObGasHum_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasHum);

template <class Archive>
void RPCObGasHum::Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("unixtime", unixtime);
    ar & boost::serialization::make_nvp("value", value);
    ar & boost::serialization::make_nvp("dpid", dpid);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasHum::Item);

template <class Archive>
void RPCObGasMix::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObGasMix-rpc", ObGasMix_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasMix);

template <class Archive>
void RPCObGasMix::Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("unixtime", unixtime);
    ar & boost::serialization::make_nvp("gas1", gas1);
    ar & boost::serialization::make_nvp("gas2", gas2);
    ar & boost::serialization::make_nvp("gas3", gas3);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasMix::Item);

template <class Archive>
void RPCObGasmap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObGasMap-rpc", ObGasMap_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasmap);

template <class Archive>
void RPCObGasmap::GasMap_Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("region", region);
    ar & boost::serialization::make_nvp("ring", ring);
    ar & boost::serialization::make_nvp("station", station);
    ar & boost::serialization::make_nvp("sector", sector);
    ar & boost::serialization::make_nvp("layer", layer);
    ar & boost::serialization::make_nvp("subsector", subsector);
    ar & boost::serialization::make_nvp("suptype", suptype);
}
COND_SERIALIZATION_INSTANTIATE(RPCObGasmap::GasMap_Item);

template <class Archive>
void RPCObImon::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObImon-rpc", ObImon_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObImon);

template <class Archive>
void RPCObImon::I_Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("value", value);
    ar & boost::serialization::make_nvp("day", day);
    ar & boost::serialization::make_nvp("time", time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObImon::I_Item);

template <class Archive>
void RPCObPVSSmap::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObIDMap-rpc", ObIDMap_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObPVSSmap);

template <class Archive>
void RPCObPVSSmap::Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("since", since);
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("region", region);
    ar & boost::serialization::make_nvp("ring", ring);
    ar & boost::serialization::make_nvp("station", station);
    ar & boost::serialization::make_nvp("sector", sector);
    ar & boost::serialization::make_nvp("layer", layer);
    ar & boost::serialization::make_nvp("subsector", subsector);
    ar & boost::serialization::make_nvp("suptype", suptype);
}
COND_SERIALIZATION_INSTANTIATE(RPCObPVSSmap::Item);

template <class Archive>
void RPCObStatus::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObStatus-rpc", ObStatus_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObStatus);

template <class Archive>
void RPCObStatus::S_Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("value", value);
    ar & boost::serialization::make_nvp("day", day);
    ar & boost::serialization::make_nvp("time", time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObStatus::S_Item);

template <class Archive>
void RPCObTemp::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObTemp-rpc", ObTemp_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObTemp);

template <class Archive>
void RPCObTemp::T_Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("value", value);
    ar & boost::serialization::make_nvp("day", day);
    ar & boost::serialization::make_nvp("time", time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObTemp::T_Item);

template <class Archive>
void RPCObUXC::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObUXC-rpc", ObUXC_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObUXC);

template <class Archive>
void RPCObUXC::Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("temperature", temperature);
    ar & boost::serialization::make_nvp("pressure", pressure);
    ar & boost::serialization::make_nvp("dewpoint", dewpoint);
    ar & boost::serialization::make_nvp("unixtime", unixtime);
}
COND_SERIALIZATION_INSTANTIATE(RPCObUXC::Item);

template <class Archive>
void RPCObVmon::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("ObVmon-rpc", ObVmon_rpc);
}
COND_SERIALIZATION_INSTANTIATE(RPCObVmon);

template <class Archive>
void RPCObVmon::V_Item::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("value", value);
    ar & boost::serialization::make_nvp("day", day);
    ar & boost::serialization::make_nvp("time", time);
}
COND_SERIALIZATION_INSTANTIATE(RPCObVmon::V_Item);

template <class Archive>
void RPCReadOutMapping::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theFeds", theFeds);
    ar & boost::serialization::make_nvp("theVersion", theVersion);
}
COND_SERIALIZATION_INSTANTIATE(RPCReadOutMapping);

template <class Archive>
void RPCStripNoises::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("v-noises", v_noises);
    ar & boost::serialization::make_nvp("v-cls", v_cls);
}
COND_SERIALIZATION_INSTANTIATE(RPCStripNoises);

template <class Archive>
void RPCStripNoises::NoiseItem::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("dpid", dpid);
    ar & boost::serialization::make_nvp("noise", noise);
    ar & boost::serialization::make_nvp("eff", eff);
    ar & boost::serialization::make_nvp("time", time);
}
COND_SERIALIZATION_INSTANTIATE(RPCStripNoises::NoiseItem);

template <class Archive>
void RPCTechTriggerConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-runId", m_runId);
    ar & boost::serialization::make_nvp("m-runType", m_runType);
    ar & boost::serialization::make_nvp("m-triggerMode", m_triggerMode);
}
COND_SERIALIZATION_INSTANTIATE(RPCTechTriggerConfig);

template <class Archive>
void TTUBoardSpecs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("m-boardspecs", m_boardspecs);
}
COND_SERIALIZATION_INSTANTIATE(TTUBoardSpecs);

template <class Archive>
void TTUBoardSpecs::TTUBoardConfig::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("RPCTechTriggerConfig", boost::serialization::base_object<RPCTechTriggerConfig>(*this));
    ar & boost::serialization::make_nvp("m-Firmware", m_Firmware);
    ar & boost::serialization::make_nvp("m-LengthOfFiber", m_LengthOfFiber);
    ar & boost::serialization::make_nvp("m-Delay", m_Delay);
    ar & boost::serialization::make_nvp("m-MaxNumWheels", m_MaxNumWheels);
    ar & boost::serialization::make_nvp("m-Wheel1Id", m_Wheel1Id);
    ar & boost::serialization::make_nvp("m-Wheel2Id", m_Wheel2Id);
    ar & boost::serialization::make_nvp("m-TrackLength", m_TrackLength);
    ar & boost::serialization::make_nvp("m-MaskedSectors", m_MaskedSectors);
    ar & boost::serialization::make_nvp("m-ForcedSectors", m_ForcedSectors);
    ar & boost::serialization::make_nvp("m-LogicType", m_LogicType);
}
COND_SERIALIZATION_INSTANTIATE(TTUBoardSpecs::TTUBoardConfig);

template <class Archive>
void TriggerBoardSpec::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("theNum", theNum);
    ar & boost::serialization::make_nvp("theMaskedLinks", theMaskedLinks);
    ar & boost::serialization::make_nvp("theLinks", theLinks);
}
COND_SERIALIZATION_INSTANTIATE(TriggerBoardSpec);

