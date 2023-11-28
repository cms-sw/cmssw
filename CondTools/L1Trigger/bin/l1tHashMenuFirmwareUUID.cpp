#include <iostream>
#include <string>

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

void showHelpMessage() {
  std::cout << "---------------------------------------------------------------\n";
  std::cout << "===                 l1tHashMenuFirmwareUUID                 ===\n";
  std::cout << "=== compute 32-bit hashed version of L1T-menu firmware UUID ===\n";
  std::cout << "---------------------------------------------------------------\n";
  std::cout << "\n"
            << "Purpose:\n"
            << "  return 32-bit hashed version (type: int) of the firmware-UUID of a L1T menu\n\n"
            << "Input:\n"
            << "  firmware-UUID of a L1T menu, i.e. value of the field \"uuid-firmware\""
            << " in the .xml file containing the L1T menu;\n"
            << "  given a database payload XYZ, the .xml file can be obtained via \"conddb dump XYZ > tmp.xml\"\n"
            << "  (if \"-h\" or \"--help\" are specified, this help message is shown)\n\n"
            << "Exit code:\n"
            << "  1 if no command-line arguments are specified, 0 otherwise\n\n"
            << "Example:\n"
            << "  > l1tHashMenuFirmwareUUID 7a1a9c0b-5e34-4c25-804f-2ae8094c4832\n\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "ERROR: no L1T-menu firmware UUID specified (hint: specify --help for more info).\n";
    return 1;
  } else {
    for (int idx = 1; idx < argc; ++idx) {
      std::string const argv_i = argv[idx];
      if (argv_i == "-h" or argv_i == "--help") {
        showHelpMessage();
        return 0;
      }
    }

    if (argc > 2) {
      std::cerr << "WARNING: specified " << argc - 1 << " command-line arguments,"
                << " but only the first one will be used (" << argv[1] << ").\n";
    }
  }

  L1TUtmTriggerMenu foo;
  foo.setFirmwareUuid(argv[1]);

  std::cout << int(foo.getFirmwareUuidHashed()) << std::endl;

  return 0;
}
