// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     ControllerChannel
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files
#include <cassert>

// user include files
#include "FWCore/SharedMemory/interface/ControllerChannel.h"
#include "FWCore/SharedMemory/interface/channel_names.h"

//
// constants, enums and typedefs
//
using namespace edm::shared_memory;
using namespace boost::interprocess;

//
// static data member definitions
//

//
// constructors and destructor
//

ControllerChannel::ControllerChannel(std::string const& iName, int id, unsigned int iMaxWaitInSeconds)
    : id_{id},
      maxWaitInSeconds_{iMaxWaitInSeconds},
      smName_{uniqueName(iName)},
      managed_sm_{open_or_create, smName_.c_str(), 1024},
      toWorkerBufferInfo_{bufferInfo(channel_names::kToWorkerBufferInfo, managed_sm_)},
      fromWorkerBufferInfo_{bufferInfo(channel_names::kFromWorkerBufferInfo, managed_sm_)},
      mutex_{open_or_create, uniqueName(channel_names::kMutex).c_str()},
      cndFromMain_{open_or_create, uniqueName(channel_names::kConditionFromMain).c_str()},
      cndToMain_{open_or_create, uniqueName(channel_names::kConditionToMain).c_str()} {
  managed_sm_.destroy<bool>(channel_names::kStop);
  stop_ = managed_sm_.construct<bool>(channel_names::kStop)(false);
  assert(stop_);
  keepEvent_ = managed_sm_.construct<bool>(channel_names::kKeepEvent)(true);
  assert(keepEvent_);

  managed_sm_.destroy<edm::Transition>(channel_names::kTransitionType);
  transitionType_ =
      managed_sm_.construct<edm::Transition>(channel_names::kTransitionType)(edm::Transition::NumberOfTransitions);
  assert(transitionType_);

  managed_sm_.destroy<unsigned long long>(channel_names::kTransitionID);
  transitionID_ = managed_sm_.construct<unsigned long long>(channel_names::kTransitionID)(0);
  assert(transitionID_);
}

ControllerChannel::~ControllerChannel() {
  managed_sm_.destroy<bool>(channel_names::kKeepEvent);
  managed_sm_.destroy<bool>(channel_names::kStop);
  managed_sm_.destroy<unsigned int>(channel_names::kTransitionType);
  managed_sm_.destroy<unsigned long long>(channel_names::kTransitionID);
  managed_sm_.destroy<BufferInfo>(channel_names::kToWorkerBufferInfo);
  managed_sm_.destroy<BufferInfo>(channel_names::kFromWorkerBufferInfo);

  named_mutex::remove(uniqueName(channel_names::kMutex).c_str());
  named_condition::remove(uniqueName(channel_names::kConditionFromMain).c_str());
  named_condition::remove(uniqueName(channel_names::kConditionToMain).c_str());
}

//
// member functions
//
std::string ControllerChannel::uniqueName(std::string iBase) const {
  auto pid = getpid();
  iBase += std::to_string(pid);
  iBase += "_";
  iBase += std::to_string(id_);

  return iBase;
}

bool ControllerChannel::wait(scoped_lock<named_mutex>& lock, edm::Transition iTrans, unsigned long long iTransID) {
  *transitionType_ = iTrans;
  *transitionID_ = iTransID;
  //std::cout << id_ << " notifying" << std::endl;
  cndFromMain_.notify_all();

  //std::cout << id_ << " waiting" << std::endl;
  using namespace boost::posix_time;
  if (not cndToMain_.timed_wait(lock, microsec_clock::universal_time() + seconds(maxWaitInSeconds_))) {
    //std::cout << id_ << " waiting FAILED" << std::endl;
    return false;
  }
  return true;
}

//
// const member functions
//

//
// static member functions
//
BufferInfo* ControllerChannel::bufferInfo(const char* iWhich, managed_shared_memory& mem) {
  mem.destroy<BufferInfo>(iWhich);
  BufferInfo* v = mem.construct<BufferInfo>(iWhich)();
  return v;
}
