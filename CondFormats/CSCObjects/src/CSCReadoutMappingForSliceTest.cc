#include <CondFormats/CSCObjects/interface/CSCReadoutMappingForSliceTest.h>
#include <iostream>
#include <fstream>
#include <sstream>

CSCReadoutMappingForSliceTest::CSCReadoutMappingForSliceTest() {}

CSCReadoutMappingForSliceTest::~CSCReadoutMappingForSliceTest() {}

int CSCReadoutMappingForSliceTest::hwId(int endcap, int station, int vmecrate, int dmb, int tmb) const {
  int id = 0;

  // This is ONLY for Slice Test Nov-2005 and onwards...

  id = vmecrate * 16 + dmb;

  // LogDebug("CSC") << " hardware id for endcap " << endcap <<
  // " station " << station << " vmecrate " << vmecrate << " dmb slot " << dmb <<
  // " tmb slot " << tmb << " = " << id;
  return id;
}
