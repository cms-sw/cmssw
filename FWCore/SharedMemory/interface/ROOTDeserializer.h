#ifndef FWCore_SharedMemory_ROOTDeserializer_h
#define FWCore_SharedMemory_ROOTDeserializer_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     ROOTDeserializer
//
/**\class ROOTDeserializer ROOTDeserializer.h " FWCore/SharedMemory/interface/ROOTDeserializer.h"

 Description: Use ROOT dictionaries to deserialize object from a buffer

 Usage:
    Used in conjunction with ROOTSerializer.

*/
//
// Original Author:  Chris Jones
//         Created:  22/01/2020
//

// system include files
#include "TClass.h"
#include "TBufferFile.h"

// user include files

// forward declarations

namespace edm::shared_memory {
  template <typename T, typename READBUFFER>
  class ROOTDeserializer {
  public:
    ROOTDeserializer(READBUFFER& iBuffer)
        : buffer_{iBuffer}, class_{TClass::GetClass(typeid(T))}, bufferFile_(TBuffer::kRead) {}

    ROOTDeserializer(const ROOTDeserializer&) = delete;
    const ROOTDeserializer& operator=(const ROOTDeserializer&) = delete;
    ROOTDeserializer(ROOTDeserializer&&) = delete;
    const ROOTDeserializer& operator=(ROOTDeserializer&&) = delete;

    // ---------- const member functions ---------------------

    // ---------- member functions ---------------------------
    T deserialize() {
      T value;
      if (buffer_.mustGetBufferAgain()) {
        auto buff = buffer_.buffer();
        bufferFile_.SetBuffer(buff.first, buff.second, kFALSE);
      }

      class_->ReadBuffer(bufferFile_, &value);
      bufferFile_.Reset();
      return value;
    }

  private:
    // ---------- member data --------------------------------
    READBUFFER& buffer_;
    TClass* const class_;
    TBufferFile bufferFile_;
  };
}  // namespace edm::shared_memory

#endif
