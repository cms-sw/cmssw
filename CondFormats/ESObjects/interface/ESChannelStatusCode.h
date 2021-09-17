#ifndef CondFormats_ESObjects_ESChannelStatusCode_H
#define CondFormats_ESObjects_ESChannelStatusCode_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>
#include <cstdint>

class ESChannelStatusCode {
public:
  ESChannelStatusCode();
  ESChannelStatusCode(const ESChannelStatusCode& codeStatus);
  ESChannelStatusCode(const uint16_t& encodedStatus) : status_(encodedStatus){};
  ~ESChannelStatusCode();

  //get Methods to be defined according to the final definition

  void print(std::ostream& s) const { s << "status is: " << status_; }

  ESChannelStatusCode& operator=(const ESChannelStatusCode& rhs);
  uint16_t getStatusCode() const { return status_; }

private:
  uint16_t status_;

  COND_SERIALIZABLE;
};
#endif
