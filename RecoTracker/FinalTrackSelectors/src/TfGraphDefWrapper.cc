#include "RecoTracker/FinalTrackSelectors/interface/TfGraphDefWrapper.h"

TfGraphDefWrapper::TfGraphDefWrapper(tensorflow::Session* session, tensorflow::GraphDef* graph)
    : session_(session), graph_(graph) {}
const tensorflow::Session* TfGraphDefWrapper::getSession() const {
  return const_cast<const tensorflow::Session*>(session_);
}

TfGraphDefWrapper::~TfGraphDefWrapper() {
  tensorflow::closeSession(session_);
  delete graph_;
};
