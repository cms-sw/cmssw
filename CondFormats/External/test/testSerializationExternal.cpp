#include "CondFormats/Serialization/interface/Test.h"

#include "CondFormats/External/interface/CLHEP.h"
#include "CondFormats/External/interface/DetID.h"
#include "CondFormats/External/interface/EcalDetID.h"
#include "CondFormats/External/interface/HLTPrescaleTable.h"
#include "CondFormats/External/interface/L1GtLogicParser.h"
#include "CondFormats/External/interface/SMatrix.h"
#include "CondFormats/External/interface/Timestamp.h"
#include "CondFormats/External/interface/PixelFEDChannel.h"

int main() {
  testSerialization<DetId>();
  testSerialization<EBDetId>();
  testSerialization<EcalContainer<EBDetId, float>>();
  testSerialization<trigger::HLTPrescaleTable>();
  testSerialization<L1GtLogicParser::TokenRPN>();
  testSerialization<edm::Timestamp>();
  testSerialization<CLHEP::Hep3Vector>();
  testSerialization<CLHEP::HepEulerAngles>();
  testSerialization<ROOT::Math::SMatrix<double, 2, 3>>();
  testSerialization<PixelFEDChannel>();

  return 0;
}
