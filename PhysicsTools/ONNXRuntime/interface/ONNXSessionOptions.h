#ifndef PHYSICSTOOLS_ONNXRUNTIME_ONNXSESSIONOPTIONS_H 
#define PHYSICSTOOLS_ONNXRUNTIME_ONNXSESSIONOPTIONS_H 

#include "ONNXRuntime.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <string>

namespace cms::Ort {

  // param_backend
  //    cpu -> Use CPU backend
  //    cuda -> Use cuda backend
  //    default -> Use best available
  inline ::Ort::SessionOptions getSessionOptions(const std::string &param_backend ) {
    
    auto backend = cms::Ort::Backend::cpu;
    if ( param_backend == "cuda" )
      backend = cms::Ort::Backend::cuda;
    
    return ONNXRuntime::defaultSessionOptions(backend);
  }
}

#endif
