// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     WorkerChannel
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
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/SharedMemory/interface/channel_names.h"

//
// constants, enums and typedefs
//
using namespace edm::shared_memory;
using namespace boost::interprocess;

namespace {
  std::string unique_name(std::string iBase, std::string_view ID) {
    iBase.append(ID);
    return iBase;
  }
}  // namespace
//
// static data member definitions
//

//
// constructors and destructor
//

WorkerChannel::WorkerChannel(std::string const& iName, const std::string& iUniqueID)
    : managed_shm_{open_only, iName.c_str()},
      mutex_{open_only, unique_name(channel_names::kMutex, iUniqueID).c_str()},
      cndFromController_{open_only, unique_name(channel_names::kConditionFromMain, iUniqueID).c_str()},
      stop_{managed_shm_.find<bool>(channel_names::kStop).first},
      transitionType_{managed_shm_.find<edm::Transition>(channel_names::kTransitionType).first},
      transitionID_{managed_shm_.find<unsigned long long>(channel_names::kTransitionID).first},
      toWorkerBufferInfo_{managed_shm_.find<BufferInfo>(channel_names::kToWorkerBufferInfo).first},
      fromWorkerBufferInfo_{managed_shm_.find<BufferInfo>(channel_names::kFromWorkerBufferInfo).first},
      cndToController_{open_only, unique_name(channel_names::kConditionToMain, iUniqueID).c_str()},
      keepEvent_{managed_shm_.find<bool>(channel_names::kKeepEvent).first},
      lock_{mutex_} {
  assert(stop_);
  assert(transitionType_);
  assert(transitionID_);
  assert(toWorkerBufferInfo_);
  assert(fromWorkerBufferInfo_);
}

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
