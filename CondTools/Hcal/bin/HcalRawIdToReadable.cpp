#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: HcalRawIdToReadable [0xhex] | [decimal]\n";
  } else {
    long j;
    for (int i = 1; i < argc; i++) {
      j = strtol(argv[i], nullptr, 0);
      std::cout << " '" << argv[i] << "' ("
                << "0x" << std::hex << j << std::dec << ", " << j << ") = " << HcalGenericDetId(j) << std::endl;
    }
  }
  return 0;
}
