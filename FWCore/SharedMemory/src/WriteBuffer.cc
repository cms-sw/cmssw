// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     WriteBuffer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files

// user include files
#include "FWCore/SharedMemory/interface/WriteBuffer.h"

//
// constants, enums and typedefs
//
using namespace edm::shared_memory;

//
// static data member definitions
//

//
// constructors and destructor
//

WriteBuffer::~WriteBuffer() {
  if (sm_) {
    sm_->destroy<char>(buffer_names::kBuffer);
    sm_.reset();
    boost::interprocess::shared_memory_object::remove(bufferNames_[*bufferIndex_].c_str());
  }
}
//
// member functions
//
void WriteBuffer::growBuffer(std::size_t iLength) {
  int newBuffer = (*bufferIndex_ + 1) % 2;
  if (sm_) {
    sm_->destroy<char>(buffer_names::kBuffer);
    sm_.reset();
    boost::interprocess::shared_memory_object::remove(bufferNames_[*bufferIndex_].c_str());
  }
  sm_ = std::make_unique<boost::interprocess::managed_shared_memory>(
      boost::interprocess::open_or_create, bufferNames_[newBuffer].c_str(), iLength + 1024);
  assert(sm_.get());
  bufferSize_ = iLength;
  *bufferIndex_ = newBuffer;
  buffer_ = sm_->construct<char>(buffer_names::kBuffer)[iLength](0);
  assert(buffer_);
}

//
// const member functions
//

//
// static member functions
//
