#ifndef CondFormats_EcalObjects_throwInvalidRawIdException_h
#define CondFormats_EcalObjects_throwInvalidRawIdException_h
// -*- C++ -*-
//
// Package:     CondFormats/EcalObjects
// Class  :     throwInvalidRawIdException
// 
/**\function ecalobjects::throwInvalidRawIdException throwInvalidRawIdException.h "CondFormats/EcalObjects/interface/throwInvalidRawIdException.h"

 Description: Function to throw an invalid index exception

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 11 Jun 2014 20:37:28 GMT
//

// system include files

// user include files

// forward declarations
namespace ecalobjects {
  void throwInvalidRawIdException(const char* iAction, uint32_t iBadID);
}

#endif
