#ifndef PhysicsTools_TensorFlow_TfGraphDefWrapper_h
#define PhysicsTools_TensorFlow_TfGraphDefWrapper_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class TfGraphDefWrapper {
public:
  TfGraphDefWrapper(tensorflow::Session*, tensorflow::GraphDef*);
  ~TfGraphDefWrapper();
  TfGraphDefWrapper(const TfGraphDefWrapper&) = delete;
  TfGraphDefWrapper& operator=(const TfGraphDefWrapper&) = delete;
  TfGraphDefWrapper(TfGraphDefWrapper&&) = delete;
  TfGraphDefWrapper& operator=(TfGraphDefWrapper&&) = delete;
  const tensorflow::Session* getSession() const;

private:
  tensorflow::Session* session_;
  std::unique_ptr<tensorflow::GraphDef> graph_;
};

#endif
