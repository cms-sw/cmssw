#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/Calibration/src/headers.h"

int main() {
  testSerialization<Algo>();
  testSerialization<AlgoMap>();
  testSerialization<Algob>();
  testSerialization<BlobComplex>();
  testSerialization<BlobComplexContent>();
  testSerialization<BlobComplexContent::Data>();
  testSerialization<BlobComplexData>();
  testSerialization<BlobComplexObjects>();
  testSerialization<BlobNoises>();
  testSerialization<BlobNoises::DetRegistry>();
  testSerialization<BlobPedestals>();
  testSerialization<CalibHistogram>();
  testSerialization<CalibHistograms>();
  testSerialization<Pedestals>();
  testSerialization<Pedestals::Item>();
  testSerialization<big>();
  testSerialization<big::bigEntry>();
  testSerialization<big::bigHeader>();
  testSerialization<big::bigStore>();
  testSerialization<boostTypeObj>();
  testSerialization<child>();
  testSerialization<condex::ConfF>();
  testSerialization<condex::ConfI>();
  //testSerialization<condex::Efficiency>(); abstract
  testSerialization<condex::ParametricEfficiencyInEta>();
  testSerialization<condex::ParametricEfficiencyInPt>();
  testSerialization<fakeMenu>();
  testSerialization<fixedArray<unsigned short, 2097>>();
  testSerialization<mySiStripNoises>();
  testSerialization<mySiStripNoises::DetRegistry>();
  testSerialization<mybase>();
  testSerialization<mypt>();
  testSerialization<std::map<std::string, Algo>>();
  testSerialization<std::map<std::string, Algob>>();
  testSerialization<std::pair<std::string, Algo>>();
  testSerialization<std::pair<std::string, Algob>>();
  testSerialization<std::vector<BlobComplexContent>>();
  testSerialization<std::vector<BlobComplexObjects>>();
  testSerialization<std::vector<BlobNoises::DetRegistry>>();
  testSerialization<std::vector<CalibHistogram>>();
  testSerialization<std::vector<Pedestals::Item>>();
  //testSerialization<std::vector<Pedestals::Item>::const_iterator>(); iterator
  testSerialization<std::vector<big::bigEntry>>();
  testSerialization<std::vector<big::bigHeader>>();
  testSerialization<std::vector<big::bigStore>>();
  testSerialization<std::vector<mySiStripNoises::DetRegistry>>();
  testSerialization<strKeyMap>();

  return 0;
}
