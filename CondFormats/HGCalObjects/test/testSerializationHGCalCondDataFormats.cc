#include "CondFormats/Serialization/interface/Test.h"
#include "CondFormats/HGCalObjects/src/headers.h"

int main() {
  //dense indexers
  testSerialization<HGCalDenseIndexerBase>();
  testSerialization<HGCalMappingCellIndexer>();
  testSerialization<HGCalFEDReadoutSequence_t>();
  testSerialization<HGCalMappingModuleIndexer>();

  return 0;
}
