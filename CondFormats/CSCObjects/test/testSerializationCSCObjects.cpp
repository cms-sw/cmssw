#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/CSCObjects/src/headers.h"

int main() {
  testSerialization<CSCBadChambers>();
  testSerialization<CSCBadStrips>();
  testSerialization<CSCBadStrips::BadChamber>();
  testSerialization<CSCBadStrips::BadChannel>();
  testSerialization<CSCBadWires>();
  testSerialization<CSCBadWires::BadChamber>();
  testSerialization<CSCBadWires::BadChannel>();
  testSerialization<CSCChamberIndex>();
  testSerialization<CSCChamberMap>();
  testSerialization<CSCChamberTimeCorrections>();
  testSerialization<CSCChamberTimeCorrections::ChamberTimeCorrections>();
  testSerialization<CSCCrateMap>();
  testSerialization<CSCDBChipSpeedCorrection>();
  testSerialization<CSCDBChipSpeedCorrection::Item>();
  testSerialization<CSCDBCrosstalk>();
  testSerialization<CSCDBCrosstalk::Item>();
  testSerialization<CSCDBGains>();
  testSerialization<CSCDBGains::Item>();
  testSerialization<CSCDBGasGainCorrection>();
  testSerialization<CSCDBGasGainCorrection::Item>();
  testSerialization<CSCDBL1TPParameters>();
  testSerialization<CSCDBNoiseMatrix>();
  testSerialization<CSCDBNoiseMatrix::Item>();
  testSerialization<CSCDBPedestals>();
  testSerialization<CSCDBPedestals::Item>();
  testSerialization<CSCDDUMap>();
  testSerialization<CSCGains>();
  testSerialization<CSCGains::Item>();
  testSerialization<CSCIdentifier>();
  testSerialization<CSCIdentifier::Item>();
  testSerialization<CSCL1TPParameters>();
  testSerialization<CSCMapItem>();
  testSerialization<CSCMapItem::MapItem>();
  testSerialization<CSCNoiseMatrix>();
  testSerialization<CSCNoiseMatrix::Item>();
  testSerialization<CSCPedestals>();
  testSerialization<CSCPedestals::Item>();
  //testSerialization<CSCReadoutMapping>(); abstract
  testSerialization<CSCReadoutMapping::CSCLabel>();
  //testSerialization<CSCTriggerMapping>(); abstract
  testSerialization<CSCTriggerMapping::CSCTriggerConnection>();
  testSerialization<CSCcrosstalk>();
  testSerialization<CSCcrosstalk::Item>();
  testSerialization<cscdqm::DCSAddressType>();
  testSerialization<cscdqm::DCSData>();
  testSerialization<cscdqm::HVVMeasType>();
  testSerialization<cscdqm::LVIMeasType>();
  testSerialization<cscdqm::LVVMeasType>();
  testSerialization<cscdqm::TempMeasType>();
  testSerialization<std::map<int, CSCMapItem::MapItem>>();
  testSerialization<std::map<int, std::vector<CSCGains::Item>>>();
  testSerialization<std::map<int, std::vector<CSCNoiseMatrix::Item>>>();
  testSerialization<std::map<int, std::vector<CSCPedestals::Item>>>();
  testSerialization<std::map<int, std::vector<CSCcrosstalk::Item>>>();
  testSerialization<std::pair<const int, CSCMapItem::MapItem>>();
  testSerialization<std::vector<CSCChamberTimeCorrections::ChamberTimeCorrections>>();
  testSerialization<std::vector<CSCMapItem::MapItem>>();
  testSerialization<std::vector<CSCBadStrips::BadChamber>>();
  testSerialization<std::vector<CSCBadStrips::BadChannel>>();
  testSerialization<std::vector<CSCBadWires::BadChamber>>();
  testSerialization<std::vector<CSCBadWires::BadChannel>>();
  testSerialization<std::vector<CSCDBChipSpeedCorrection::Item>>();
  testSerialization<std::vector<CSCDBCrosstalk::Item>>();
  testSerialization<std::vector<CSCDBGains::Item>>();
  testSerialization<std::vector<CSCDBGasGainCorrection::Item>>();
  testSerialization<std::vector<CSCDBNoiseMatrix::Item>>();
  testSerialization<std::vector<CSCDBPedestals::Item>>();
  testSerialization<std::vector<CSCGains::Item>>();
  testSerialization<std::vector<CSCNoiseMatrix::Item>>();
  testSerialization<std::vector<CSCPedestals::Item>>();
  testSerialization<std::vector<CSCcrosstalk::Item>>();
  testSerialization<std::vector<cscdqm::DCSAddressType>>();
  testSerialization<std::vector<cscdqm::DCSData>>();
  testSerialization<std::vector<cscdqm::HVVMeasType>>();
  testSerialization<std::vector<cscdqm::LVIMeasType>>();
  testSerialization<std::vector<cscdqm::LVVMeasType>>();
  testSerialization<std::vector<cscdqm::TempMeasType>>();

  return 0;
}
