#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"

TfGraphDefWrapper::TfGraphDefWrapper(tensorflow::Session* session, tensorflow::GraphDef* graph)
    : session_(session), graph_(graph) {}
const tensorflow::Session* TfGraphDefWrapper::getSession() const { return session_; }

TfGraphDefWrapper::~TfGraphDefWrapper() { tensorflow::closeSession(session_); };
