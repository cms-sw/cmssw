#ifndef RecoTracker_FinalTrackSelectors_TfGraphDefWrapper_h
#define RecoTracker_FinalTrackSelectors_TfGraphDefWrapper_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class TfGraphDefWrapper {
public:
  TfGraphDefWrapper(tensorflow::Session*);
  ~TfGraphDefWrapper() { tensorflow::closeSession(session_); };
  const tensorflow::Session* getSession() const;

private:
  tensorflow::Session* session_;
};

#endif
