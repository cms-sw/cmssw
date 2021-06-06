#include "FWCore/Utilities/src/Guid.h"

#include <string>
#include <cassert>
#include <iostream>

int main() {
  edm::Guid guid;

  auto guidString = guid.toString();
  edm::Guid guid2(guidString, false);
  edm::Guid guid3(guid2);

  auto guidBinary = guid.toBinary();
  edm::Guid guid4(guidBinary, true);

  assert(guid == guid2);
  assert(guid == guid3);
  assert(guid == guid4);

  auto guidString2 = guid2.toString();
  auto guidString3 = guid3.toString();
  auto guidString4 = guid4.toString();

  assert(guidString2 == guidString);
  assert(guidString3 == guidString);
  assert(guidString4 == guidString);

  auto guidBinary2 = guid2.toBinary();
  auto guidBinary3 = guid3.toBinary();
  auto guidBinary4 = guid4.toBinary();

  assert(guidBinary2 == guidBinary);
  assert(guidBinary3 == guidBinary);
  assert(guidBinary4 == guidBinary);

  edm::Guid otherGuid;
  assert(otherGuid != guid);

  edm::Guid otherBinaryGuid{otherGuid.toBinary(), true};
  assert(otherBinaryGuid == otherGuid);
  assert(otherBinaryGuid != guid4);
}
