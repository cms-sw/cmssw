/**
  * \file
  * A test for EEDetId::hashedIndex()
  *
  */

#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "DataFormats/EcalDetId/interface/EcalScDetId.h"

int main(int argc, char* argv[]) {
  const char* filename = "sc.txt";
  std::ofstream out(filename);
  try {
    out << "iX\tiY\tiZ\thashed\tERR\tERR\tiX\tiY\tiZ\n";
    for (int iZ = -1; iZ <= +1; iZ += 2) {
      for (int iY = EcalScDetId::IY_MIN; iY <= EcalScDetId::IY_MAX; ++iY) {
        for (int iX = EcalScDetId::IX_MIN; iX <= EcalScDetId::IX_MAX; ++iX) {
          if (!EcalScDetId::validDetId(iX, iY, iZ))
            continue;
          EcalScDetId sc1(iX, iY, iZ);
          int ih = sc1.hashedIndex();
          out << iX << "\t" << iY << "\t" << iZ << "\t" << ih;
          EcalScDetId sc2 = EcalScDetId::unhashIndex(ih);
          out << "\t" << (sc1.rawId() == sc2.rawId() ? "OK" : "ERROR");
          out << "\t" << (sc1 == sc2 ? "OK" : "ERROR");
          out << "\t" << sc2.ix() << "\t" << sc2.iy() << "\t" << sc2.zside();
          out << "\n";
        }
      }
    }
    std::cout << "Supercystal indices have been dumped in file " << filename << std::endl;
  } catch (std::exception& e) {
    std::cerr << e.what();
  }
}
