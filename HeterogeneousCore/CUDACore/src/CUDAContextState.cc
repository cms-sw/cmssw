#include "HeterogeneousCore/CUDACore/interface/CUDAContextState.h"
#include "FWCore/Utilities/interface/Exception.h"

void CUDAContextState::throwIfStream() const {
  if(stream_) {
    throw cms::Exception("LogicError") << "Trying to set CUDAContextState, but it already had a valid state";
  }
}

void CUDAContextState::throwIfNoStream() const {
  if(not stream_) {
    throw cms::Exception("LogicError") << "Trying to get CUDAContextState, but it did not have a valid state";
  }
}
