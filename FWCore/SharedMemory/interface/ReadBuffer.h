#ifndef FWCore_SharedMemory_ReadBuffer_h
#define FWCore_SharedMemory_ReadBuffer_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     ReadBuffer
//
/**\class ReadBuffer ReadBuffer.h " FWCore/SharedMemory/interface/ReadBuffer.h"

 Description: Manages a shared memory buffer used for reading

 Usage:
      Handles reading from a dynamically allocatable shared memory buffer.
This works in conjunction with WriteBuffer.
*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files
#include <array>
#include <memory>
#include <string>
#include "boost/interprocess/managed_shared_memory.hpp"

// user include files
#include "FWCore/SharedMemory/interface/buffer_names.h"

// forward declarations

namespace edm::shared_memory {
  class ReadBuffer {
  public:
    /** iUniqueName : must be unique for all processes running on a system.
        iBufferIndex : is a pointer to a shared_memory address where the same address needs to be shared by ReadBuffer and WriteBuffer.
    */
    ReadBuffer(std::string const& iUniqueName, char* iBufferIndex)
        : buffer_{nullptr, 0}, bufferIndex_{iBufferIndex}, bufferOldIndex_{3} {
      *bufferIndex_ = 0;
      bufferNames_[0] = iUniqueName + buffer_names::kBuffer0;
      bufferNames_[1] = iUniqueName + buffer_names::kBuffer1;
    }
    ReadBuffer(const ReadBuffer&) = delete;
    const ReadBuffer& operator=(const ReadBuffer&) = delete;
    ReadBuffer(ReadBuffer&&) = delete;
    const ReadBuffer& operator=(ReadBuffer&&) = delete;

    // ---------- const member functions ---------------------
    bool mustGetBufferAgain() const { return *bufferIndex_ != bufferOldIndex_; }

    // ---------- member functions ---------------------------
    std::pair<char*, std::size_t> buffer() {
      if (mustGetBufferAgain()) {
        using namespace boost::interprocess;
        sm_ = std::make_unique<managed_shared_memory>(open_only, bufferNames_[*bufferIndex_].c_str());
        buffer_ = sm_->find<char>(buffer_names::kBuffer);
        bufferOldIndex_ = *bufferIndex_;
      }
      return buffer_;
    }

  private:
    // ---------- member data --------------------------------
    std::pair<char*, std::size_t> buffer_;
    char* bufferIndex_;
    char bufferOldIndex_;
    std::array<std::string, 2> bufferNames_;
    std::unique_ptr<boost::interprocess::managed_shared_memory> sm_;
  };
}  // namespace edm::shared_memory

#endif
