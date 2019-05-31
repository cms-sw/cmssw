#ifndef L1TObjects_L1MuCSCTFConfiguration_h
#define L1TObjects_L1MuCSCTFConfiguration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

class L1MuCSCTFConfiguration {
private:
  std::string registers[12];

public:
  const std::string* configAsText(void) const throw() { return registers; }

  edm::ParameterSet parameters(int sp) const;

  L1MuCSCTFConfiguration& operator=(const L1MuCSCTFConfiguration& conf) {
    for (int sp = 0; sp < 12; sp++)
      registers[sp] = conf.registers[sp];
    return *this;
  }

  L1MuCSCTFConfiguration(void) {}
  L1MuCSCTFConfiguration(std::string regs[12]) {
    for (int sp = 0; sp < 12; sp++)
      registers[sp] = regs[sp];
  }
  L1MuCSCTFConfiguration(const L1MuCSCTFConfiguration& conf) {
    for (int sp = 0; sp < 12; sp++)
      registers[sp] = conf.registers[sp];
  }
  ~L1MuCSCTFConfiguration(void) {}

  /// print all the L1 CSCTF Configuration Parameters
  void print(std::ostream&) const;

  COND_SERIALIZABLE;
};

#endif
