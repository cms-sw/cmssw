#include "CondFormats/L1TObjects/interface/L1MuCSCTFConfiguration.h"
#include <iostream>
#include <cstdlib>

void L1MuCSCTFConfiguration::print(std::ostream& myStr) const {
  myStr << "\nL1 Mu CSCTF Parameters \n" << std::endl;

  for (int iSP = 0; iSP < 12; iSP++) {
    myStr << "=============================================" << std::endl;
    myStr << "Printing out Global Tag Content for SP " << iSP + 1 << std::endl;
    myStr << registers[iSP];
    myStr << "=============================================" << std::endl;
  }
}
