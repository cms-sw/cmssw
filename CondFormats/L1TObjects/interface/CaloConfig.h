///
/// \class l1t::CaloConfig
///
/// Description: Placeholder for calorimeter trigger runtime configuration
///
/// Implementation:
///
///

#ifndef CaloConfig_h
#define CaloConfig_h

#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/L1TObjects/interface/LUT.h"

namespace l1t {


  class CaloConfig {

  public:

    enum { Version = 1 };

    CaloConfig() { version_= (unsigned) Version; }
    ~CaloConfig() {}
    friend class CaloConfigHelper;

  private:
    unsigned version_;
    std::vector<unsigned> uconfig_;
    std::vector<std::string> sconfig_;

    COND_SERIALIZABLE;
  };

}// namespace
#endif
