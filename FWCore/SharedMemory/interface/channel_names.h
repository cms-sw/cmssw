#ifndef FWCore_SharedMemory_channel_names_h
#define FWCore_SharedMemory_channel_names_h
// -*- C++ -*-
//
// Package:     FWCore/SharedMemory
// Class  :     channel_names
//
/**

 Description: Shared memory names used for the ControllerChannel and WorkerChannel

 Usage:
      Internal details of ControllerChannel and WorkerChannel

*/
//
// Original Author:  Chris Jones
//         Created:  21/01/2020
//

// system include files

// user include files

// forward declarations

namespace edm::shared_memory {
  namespace channel_names {
    constexpr char const* const kToWorkerBufferIndex = "bufferIndexToWorker";
    constexpr char const* const kFromWorkerBufferIndex = "bufferIndexFromWorker";
    constexpr char const* const kMutex = "mtx";
    constexpr char const* const kConditionFromMain = "cndFromMain";
    constexpr char const* const kConditionToMain = "cndToMain";
    constexpr char const* const kStop = "stop";
    constexpr char const* const kKeepEvent = "keepEvent";
    constexpr char const* const kTransitionType = "transitionType";
    constexpr char const* const kTransitionID = "transitionID";
  };  // namespace channel_names
}  // namespace edm::shared_memory

#endif
