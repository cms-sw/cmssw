#ifndef FWCore_SharedMemory_BufferInfo_h
#define FWCore_SharedMemory_BufferInfo_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     BufferInfo
//
/**\class BufferInfo BufferInfo.h " FWCore/SharedMemory/interface/BufferInfo.h"

 Description: Information needed to manage the buffer

 Usage:
    This is an internal detail of the system.
*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files

// user include files

// forward declarations

namespace edm::shared_memory {
  struct BufferInfo {
    int identifier_;
    char index_;
  };
}  // namespace edm::shared_memory

#endif
