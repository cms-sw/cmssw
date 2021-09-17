#ifndef CondFormats_EcalObjects_EcalDQMStatusCode_H
#define CondFormats_EcalObjects_EcalDQMStatusCode_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>
#include <cstdint>

class EcalDQMStatusCode {
public:
  EcalDQMStatusCode();
  EcalDQMStatusCode(const EcalDQMStatusCode& codeStatus);
  EcalDQMStatusCode(const uint32_t& encodedStatus) : status_(encodedStatus){};
  ~EcalDQMStatusCode();

  //get Methods to be defined according to the final definition

  void print(std::ostream& s) const { s << "status is: " << status_; }

  EcalDQMStatusCode& operator=(const EcalDQMStatusCode& rhs);
  uint32_t getStatusCode() const { return status_; }

private:
  uint32_t status_;

  COND_SERIALIZABLE;
};
#endif
