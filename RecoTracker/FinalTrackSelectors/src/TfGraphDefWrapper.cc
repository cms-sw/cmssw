#include "RecoTracker/FinalTrackSelectors/interface/TfGraphDefWrapper.h"

TfGraphDefWrapper::TfGraphDefWrapper(tensorflow::Session* session) : session_(session) {}
const tensorflow::Session* TfGraphDefWrapper::getSession() const {
  return const_cast<const tensorflow::Session*>(session_);
}
