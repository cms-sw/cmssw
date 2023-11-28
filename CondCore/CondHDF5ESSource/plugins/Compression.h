#ifndef CondCore_CondHDF5ESSource_Compression_h
#define CondCore_CondHDF5ESSource_Compression_h
// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     Compression
//
/**\class Compression Compression.h "Compression.h"

 Description: Which Compression algorithm is being used

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 20 Jul 2023 14:29:25 GMT
//

// system include files

// user include files

// forward declarations

namespace cond::hdf5 {
  enum class Compression { kNone, kZLIB, kLZMA };
}

#endif
