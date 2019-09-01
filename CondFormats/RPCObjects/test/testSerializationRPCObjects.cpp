#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/RPCObjects/src/headers.h"

int main() {
  testSerialization<ChamberLocationSpec>();
  testSerialization<ChamberStripSpec>();
  testSerialization<DccSpec>();
  testSerialization<FebConnectorSpec>();
  testSerialization<FebLocationSpec>();
  //testSerialization<L1RPCConeBuilder>(); struct children do not appear in the XML
  testSerialization<L1RPCDevCoords>();
  testSerialization<L1RPCHwConfig>();
  testSerialization<LinkBoardSpec>();
  testSerialization<LinkConnSpec>();
  testSerialization<RBCBoardSpecs>();
  testSerialization<RBCBoardSpecs::RBCBoardConfig>();
  testSerialization<RPCClusterSize>();
  testSerialization<RPCClusterSize::ClusterSizeItem>();
  testSerialization<RPCDQMObject>();
  testSerialization<RPCDQMObject::DQMObjectItem>();
  testSerialization<RPCEMap>();
  testSerialization<RPCEMap::dccItem>();
  testSerialization<RPCEMap::febItem>();
  testSerialization<RPCEMap::lbItem>();
  testSerialization<RPCEMap::linkItem>();
  testSerialization<RPCEMap::tbItem>();
  testSerialization<RPCObFebmap>();
  testSerialization<RPCObFebmap::Feb_Item>();
  testSerialization<RPCObGas>();
  testSerialization<RPCObGas::Item>();
  testSerialization<RPCObGasHum>();
  testSerialization<RPCObGasHum::Item>();
  testSerialization<RPCObGasMix>();
  testSerialization<RPCObGasMix::Item>();
  testSerialization<RPCObGasmap>();
  testSerialization<RPCObGasmap::GasMap_Item>();
  testSerialization<RPCObImon>();
  testSerialization<RPCObImon::I_Item>();
  testSerialization<RPCObPVSSmap>();
  testSerialization<RPCObPVSSmap::Item>();
  testSerialization<RPCObStatus>();
  testSerialization<RPCObStatus::S_Item>();
  testSerialization<RPCObTemp>();
  testSerialization<RPCObTemp::T_Item>();
  testSerialization<RPCObUXC>();
  testSerialization<RPCObUXC::Item>();
  testSerialization<RPCObVmon>();
  testSerialization<RPCObVmon::V_Item>();
  testSerialization<RPCReadOutMapping>();
  testSerialization<RPCStripNoises>();
  testSerialization<RPCStripNoises::NoiseItem>();
  testSerialization<RPCTechTriggerConfig>();
  testSerialization<TTUBoardSpecs>();
  testSerialization<TTUBoardSpecs::TTUBoardConfig>();
  testSerialization<TriggerBoardSpec>();
  testSerialization<std::map<int, DccSpec>>();
  testSerialization<std::set<L1RPCDevCoords>>();
  testSerialization<std::vector<ChamberStripSpec>>();
  testSerialization<std::vector<FebConnectorSpec>>();
  testSerialization<std::vector<LinkBoardSpec>>();
  testSerialization<std::vector<LinkConnSpec>>();
  testSerialization<std::vector<RBCBoardSpecs::RBCBoardConfig>>();
  testSerialization<std::vector<RPCClusterSize::ClusterSizeItem>>();
  testSerialization<std::vector<RPCDQMObject::DQMObjectItem>>();
  testSerialization<std::vector<RPCEMap::dccItem>>();
  testSerialization<std::vector<RPCEMap::febItem>>();
  testSerialization<std::vector<RPCEMap::lbItem>>();
  testSerialization<std::vector<RPCEMap::linkItem>>();
  testSerialization<std::vector<RPCEMap::tbItem>>();
  testSerialization<std::vector<RPCObFebmap::Feb_Item>>();
  testSerialization<std::vector<RPCObGas::Item>>();
  testSerialization<std::vector<RPCObGasHum::Item>>();
  testSerialization<std::vector<RPCObGasMix::Item>>();
  testSerialization<std::vector<RPCObGasmap::GasMap_Item>>();
  testSerialization<std::vector<RPCObImon::I_Item>>();
  testSerialization<std::vector<RPCObPVSSmap::Item>>();
  testSerialization<std::vector<RPCObStatus::S_Item>>();
  testSerialization<std::vector<RPCObTemp::T_Item>>();
  testSerialization<std::vector<RPCObUXC::Item>>();
  testSerialization<std::vector<RPCObVmon::V_Item>>();
  testSerialization<std::vector<RPCStripNoises::NoiseItem>>();
  testSerialization<std::vector<TTUBoardSpecs::TTUBoardConfig>>();
  testSerialization<std::vector<TriggerBoardSpec>>();

  return 0;
}
