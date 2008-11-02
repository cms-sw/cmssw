#include "DetectorDescription/Base/interface/DDReadMapType.h"
#include "DetectorDescription/Base/interface/DDException.h"


namespace dddDetails {
  void errorReadMapType(const std::string & key) const throw (DDException) {
    std::string message("ReadMapType::operator[] key not found:" + key);
    throw DDException(message);    
  }
}
