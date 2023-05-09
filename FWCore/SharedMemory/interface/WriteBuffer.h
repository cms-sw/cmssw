#ifndef FWCore_SharedMemory_WriteBuffer_h
#define FWCore_SharedMemory_WriteBuffer_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     WriteBuffer
//
/**\class WriteBuffer WriteBuffer.h " FWCore/SharedMemory/interface/WriteBuffer.h"

 Description: Manages a shared memory buffer used for writing

 Usage:
    Handles writing to a shared memory buffer. The buffer size grows automatically when necessary.
 This works in conjunction with ReadBuffer.

*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files
#include <array>
#include <memory>
#include <string>
#include <cassert>
#include <algorithm>
#include "boost/interprocess/managed_shared_memory.hpp"

// user include files
#include "FWCore/SharedMemory/interface/buffer_names.h"
#include "FWCore/SharedMemory/interface/BufferInfo.h"

// forward declarations

namespace edm::shared_memory {
  class WriteBuffer {
  public:
    /** iUniqueName : must be unique for all processes running on a system.
        iBufferInfo : is a pointer to a shared_memory address where the same address needs to be shared by ReadBuffer and WriteBuffer.
    */

    WriteBuffer(std::string const& iUniqueName, BufferInfo* iBufferInfo)
        : bufferSize_{0}, buffer_{nullptr}, bufferInfo_{iBufferInfo} {
      bufferNames_[0] = iUniqueName + buffer_names::kBuffer0;
      bufferNames_[1] = iUniqueName + buffer_names::kBuffer1;
      assert(bufferInfo_);
    }
    WriteBuffer(const WriteBuffer&) = delete;
    const WriteBuffer& operator=(const WriteBuffer&) = delete;
    WriteBuffer(WriteBuffer&&) = delete;
    const WriteBuffer& operator=(WriteBuffer&&) = delete;

    ~WriteBuffer();

    // ---------- member functions ---------------------------
    void copyToBuffer(const char* iStart, std::size_t iLength) {
      if (iLength > bufferSize_) {
        growBuffer(iLength);
      }
      std::copy(iStart, iStart + iLength, buffer_);
    }

  private:
    void growBuffer(std::size_t iLength);

    // ---------- member data --------------------------------
    std::size_t bufferSize_;
    char* buffer_;
    BufferInfo* bufferInfo_;
    std::array<std::string, 2> bufferNames_;
    struct SMOwner {
      SMOwner() = default;
      SMOwner(std::string const& iName, std::size_t iLength);
      ~SMOwner();
      SMOwner& operator=(SMOwner&&);
      boost::interprocess::managed_shared_memory* operator->() { return sm_.get(); }
      boost::interprocess::managed_shared_memory* get() { return sm_.get(); }
      operator bool() const { return bool(sm_); }
      void reset();

    private:
      std::string name_;
      std::unique_ptr<boost::interprocess::managed_shared_memory> sm_;
    } sm_;
  };
}  // namespace edm::shared_memory

#endif
