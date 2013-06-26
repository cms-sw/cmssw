#include "DetectorDescription/Base/interface/DDReadMapType.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace dddDetails {
  void errorReadMapType(const std::string & key) {
    throw cms::Exception("DDException") << "ReadMapType::operator[] key not found:" << key;
  }
}
