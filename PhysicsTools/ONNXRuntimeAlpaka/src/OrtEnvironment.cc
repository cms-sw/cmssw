#include "PhysicsTools/ONNXRuntimeAlpaka/interface/OrtEnvironment.h"

namespace cms::Ort::alpakatools {

  ::Ort::Env& environment() {
    static ::Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ONNXRuntimeAlpaka");
    return env;
  }

}  // namespace cms::Ort::alpakatools
