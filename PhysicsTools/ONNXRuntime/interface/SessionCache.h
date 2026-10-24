#ifndef PhysicsTools_ONNXRuntime_interface_SessionCache_h
#define PhysicsTools_ONNXRuntime_interface_SessionCache_h

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "FWCore/Utilities/interface/StreamID.h"
#include "PhysicsTools/ONNXRuntime/interface/Backend.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace cms::Ort {

  // The ONNX Runtime sessions used by a module to run a model, meant to be held in an edm::GlobalCache.
  //
  // On the CPU all the framework streams share a single session: an ONNX Runtime session can run inferences
  // concurrently from multiple threads, while a session per stream would keep a copy of the model and of its memory
  // arena for each stream.
  // On a GPU each framework stream uses its own session, running on the device chosen by the ONNXService for that
  // stream, in a compute stream owned by the session.
  //
  // The session options can contain the settings specific to the model or to the module (e.g. the graph optimisation
  // level), but not an execution provider: they are copied for each session, and the ONNXService adds the framework
  // settings and the execution provider for the device of the framework stream.
  class SessionCache {
  public:
    SessionCache(std::string model_path,
                 Backend backend,
                 ::Ort::SessionOptions const& options = ::Ort::SessionOptions());

    SessionCache(SessionCache const&) = delete;
    SessionCache& operator=(SessionCache const&) = delete;

    Backend backend() const { return backend_; }

    // The session used by the given framework stream, created by the first call that needs it: for Backend::cpu the
    // first call from any framework stream, for a GPU backend the first call from each framework stream. Call it in
    // beginStream() to create the sessions before the first event.
    ONNXRuntime const& get(edm::StreamID id) const;

  private:
    std::unique_ptr<ONNXRuntime> createSession(edm::StreamID id) const;

    const std::string model_path_;
    const Backend backend_;
    const ::Ort::SessionOptions options_;

    // Backend::cpu: a single session, shared by all the framework streams.
    // GPU backends: the sessions for each framework stream, indexed by the stream id.
    // The mutex protects the vector, that may be resized concurrently by the other streams.
    mutable std::mutex mutex_;
    mutable std::vector<std::unique_ptr<ONNXRuntime>> sessions_;
  };

}  // namespace cms::Ort

#endif  // PhysicsTools_ONNXRuntime_interface_SessionCache_h
