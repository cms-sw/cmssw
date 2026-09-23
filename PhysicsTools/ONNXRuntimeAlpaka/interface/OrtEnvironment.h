#ifndef PhysicsTools_ONNXRuntimeAlpaka_interface_OrtEnvironment_h
#define PhysicsTools_ONNXRuntimeAlpaka_interface_OrtEnvironment_h

#include "onnxruntime/onnxruntime_cxx_api.h"

namespace cms::Ort::alpakatools {

  // Process-wide ONNX Runtime environment, shared by all the AlpakaSession objects.
  ::Ort::Env& environment();

}  // namespace cms::Ort::alpakatools

#endif  // PhysicsTools_ONNXRuntimeAlpaka_interface_OrtEnvironment_h
