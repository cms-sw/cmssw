// -*- C++ -*-
//
// Package:     CondFormats/EcalObjects
// Class  :     throwInvalidIndexException
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 11 Jun 2014 20:37:36 GMT
//

// system include files

// user include files
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/EcalObjects/interface/throwInvalidRawIdException.h"

namespace ecalobjects {
  void throwInvalidRawIdException(const char* iAction, uint32_t iBadID) {
    throw cms::Exception("InvalidRawId")<<iAction<<" used invalid rawid "<<iBadID;
  }
}
