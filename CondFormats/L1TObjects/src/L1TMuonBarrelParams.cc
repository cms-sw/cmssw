#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

void L1TMuonBarrelParams::print(std::ostream& out) const {

  out << "L1 BMTF Parameters" << std::endl;

  out << "Firmware version: " << fwVersion_ << std::endl;
}
