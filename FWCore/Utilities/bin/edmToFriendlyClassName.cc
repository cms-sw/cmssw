// -*- C++ -*-
//
// Package:     Utilities
// Class  :     edmToFriendlyClassName
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Oct  4 14:30:17 EDT 2007
//

// user include files
#include "FWCore/Utilities/interface/FriendlyName.h"

// system include files
#include <iostream>
#include <stdexcept>

int
main(int argc, char* argv[]) {
  try {
    for(int index = 1; index < argc; ++index) {
      std::cout << edm::friendlyname::friendlyName(argv[index]) << std::endl;
    }
  }
  catch(std::runtime_error const& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
