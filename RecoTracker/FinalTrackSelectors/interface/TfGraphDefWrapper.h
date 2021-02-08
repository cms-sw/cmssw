#ifndef RecoTracker_FinalTrackSelectors_TfGraphDefWrapper_h
#define RecoTracker_FinalTrackSelectors_TfGraphDefWrapper_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class TfGraphDefWrapper {
public:
  TfGraphDefWrapper(tensorflow::Session*);
  ~TfGraphDefWrapper() { tensorflow::closeSession(session_); };
  TfGraphDefWrapper(const TfGraphDefWrapper&) = delete;
  TfGraphDefWrapper& operator=(const TfGraphDefWrapper&) = delete;
  TfGraphDefWrapper(TfGraphDefWrapper&&) = delete;
  TfGraphDefWrapper& operator=(TfGraphDefWrapper&&) = delete;
  const tensorflow::Session* getSession() const;

private:
  tensorflow::Session* session_;
};

#endif
