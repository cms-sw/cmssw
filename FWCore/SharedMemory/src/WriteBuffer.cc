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
#include "FWCore/Utilities/interface/Exception.h"
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
WriteBuffer::SMOwner::SMOwner(std::string const& iName, std::size_t iLength) : name_(iName) {
  //do a remove first just in case a previous job had a failure and left a same named
  // memory object
  boost::interprocess::shared_memory_object::remove(iName.c_str());
  sm_ = std::make_unique<boost::interprocess::managed_shared_memory>(
      boost::interprocess::create_only, iName.c_str(), iLength);
}

WriteBuffer::SMOwner::~SMOwner() {
  if (sm_) {
    boost::interprocess::shared_memory_object::remove(name_.c_str());
  }
}

WriteBuffer::SMOwner& WriteBuffer::SMOwner::operator=(WriteBuffer::SMOwner&& other) {
  if (sm_) {
    boost::interprocess::shared_memory_object::remove(name_.c_str());
  }
  name_ = std::move(other.name_);
  sm_ = std::move(other.sm_);
  return *this;
}

void WriteBuffer::SMOwner::reset() { *this = SMOwner(); }

WriteBuffer::~WriteBuffer() {
  if (sm_) {
    sm_->destroy<char>(buffer_names::kBuffer);
  }
}

//
// member functions
//
void WriteBuffer::growBuffer(std::size_t iLength) {
  int newBuffer = (bufferInfo_->index_ + 1) % 2;
  bool destroyedBuffer = false;
  auto oldIndex = bufferInfo_->index_;
  if (sm_) {
    destroyedBuffer = true;
    try {
      sm_->destroy<char>(buffer_names::kBuffer);
    } catch (boost::interprocess::interprocess_exception const& iExcept) {
      throw cms::Exception("SharedMemory")
          << "in growBuffer while destroying the shared memory object the following exception was caught\n"
          << iExcept.what();
    }
    try {
      sm_.reset();
    } catch (boost::interprocess::interprocess_exception const& iExcept) {
      throw cms::Exception("SharedMemory")
          << "in growBuffer while removing the shared memory object named '" << bufferNames_[bufferInfo_->index_]
          << "' the following exception was caught\n"
          << iExcept.what();
    }
  }
  try {
    sm_ = SMOwner(bufferNames_[newBuffer], iLength + 1024);
  } catch (boost::interprocess::interprocess_exception const& iExcept) {
    throw cms::Exception("SharedMemory") << "in growBuffer while creating the shared memory object '"
                                         << bufferNames_[newBuffer] << "' of length " << iLength + 1024
                                         << " the following exception was caught\n"
                                         << iExcept.what();
  }
  assert(sm_.get());
  bufferSize_ = iLength;
  bufferInfo_->index_ = newBuffer;
  bufferInfo_->identifier_ = bufferInfo_->identifier_ + 1;
  try {
    buffer_ = sm_->construct<char>(buffer_names::kBuffer)[iLength](0);
  } catch (boost::interprocess::interprocess_exception const& iExcept) {
    cms::Exception except("SharedMemory");
    except << "boost::interprocess exception caught: " << iExcept.what();
    {
      std::ostringstream os;
      os << "in growBuffer while creating the buffer within the shared memory object '" << bufferNames_[newBuffer]
         << "' with index " << newBuffer << " where the buffer is of length " << iLength;

      if (destroyedBuffer) {
        os << " after destroying the previous shared memory object '" << bufferNames_[oldIndex] << "' with index "
           << oldIndex;
      }
      except.addContext(os.str());
    }
    throw except;
  }
  assert(buffer_);
}

//
// const member functions
//

//
// static member functions
//
