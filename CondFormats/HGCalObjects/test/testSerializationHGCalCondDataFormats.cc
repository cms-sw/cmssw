#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/HGCalObjects/interface/HGCalDenseIndexerBase.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"

int main() {
  //dense indexers
  testSerialization<HGCalDenseIndexerBase>();
  testSerialization<HGCalMappingCellIndexer>();
  testSerialization<HGCalFEDReadoutSequence_t>();
  testSerialization<HGCalMappingModuleIndexer>();

  return 0;
}
