#ifndef FWCore_SharedMemory_buffer_names_h
#define FWCore_SharedMemory_buffer_names_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     buffer_names
//
/**

 Description: Shared memory names used for the ReadBuffer and WriteBuffer

 Usage:
      Internal details of ReadBuffer and WriteBuffer

*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files

// user include files

// forward declarations

namespace edm::shared_memory {
  namespace buffer_names {
    constexpr char const* const kBuffer = "buffer";
    constexpr char const* const kBuffer0 = "buffer0";
    constexpr char const* const kBuffer1 = "buffer1";
  }  // namespace buffer_names
}  // namespace edm::shared_memory

#endif
