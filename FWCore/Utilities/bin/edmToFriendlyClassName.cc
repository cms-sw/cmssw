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
// $Id: edmToFriendlyClassName.cc,v 1.1 2007/10/04 19:23:34 chrjones Exp $
//

// system include files
#include <iostream>
// user include files
#include "FWCore/Utilities/interface/FriendlyName.h"

int
main(int argc, char* argv[])
{
  for(int index=1; index < argc; ++index) {
    std::cout <<edm::friendlyname::friendlyName(argv[index])<<std::endl;
  }
  return 0;
}
