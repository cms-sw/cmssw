#ifndef FWCore_SharedMemory_ROOTSerializer_h
#define FWCore_SharedMemory_ROOTSerializer_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     ROOTSerializer
//
/**\class ROOTSerializer ROOTSerializer.h " FWCore/SharedMemory/interface/ROOTSerializer.h"

 Description: Use ROOT dictionaries to serialize object to a buffer

 Usage:
    This is used in conjuction with ROOTDeserializer.

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
  template <typename T, typename WRITEBUFFER>
  class ROOTSerializer {
  public:
    ROOTSerializer(WRITEBUFFER& iBuffer)
        : buffer_(iBuffer), class_{TClass::GetClass(typeid(T))}, bufferFile_{TBuffer::kWrite} {}

    ROOTSerializer(const ROOTSerializer&) = delete;
    const ROOTSerializer& operator=(const ROOTSerializer&) = delete;
    ROOTSerializer(ROOTSerializer&&) = delete;
    const ROOTSerializer& operator=(ROOTSerializer&&) = delete;

    // ---------- const member functions ---------------------

    // ---------- member functions ---------------------------
    void serialize(T& iValue) {
      bufferFile_.Reset();
      class_->WriteBuffer(bufferFile_, &iValue);

      buffer_.copyToBuffer(bufferFile_.Buffer(), bufferFile_.Length());
    }

  private:
    // ---------- member data --------------------------------
    WRITEBUFFER& buffer_;
    TClass* const class_;
    TBufferFile bufferFile_;
  };
}  // namespace edm::shared_memory

#endif
