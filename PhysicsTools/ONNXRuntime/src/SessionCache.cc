#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXInterface.h"
#include "PhysicsTools/ONNXRuntime/interface/SessionCache.h"

namespace cms::Ort {

  SessionCache::SessionCache(std::string model_path, Backend backend, ::Ort::SessionOptions const& options)
      : model_path_(std::move(model_path)), backend_(backend), options_(options.Clone()) {}

  ONNXRuntime const& SessionCache::get(edm::StreamID id) const {
    const bool shared = (backend_ == Backend::cpu);
    const unsigned int index = shared ? 0 : id.value();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (index < sessions_.size() and sessions_[index]) {
        return *sessions_[index];
      }
      if (shared) {
        // Create the shared session while holding the lock: the other framework streams wait for it, instead of
        // creating their own.
        sessions_.emplace_back(createSession(id));
        return *sessions_.front();
      }
    }

    // Create the session without holding the lock, because it can take a long time (e.g. to compile the model for the
    // GPU): only this framework stream creates and stores the session with this index.
    auto session = createSession(id);
    // The session object does not move when the vector is resized.
    ONNXRuntime const* result = session.get();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (index >= sessions_.size()) {
        sessions_.resize(index + 1);
      }
      sessions_[index] = std::move(session);
    }
    return *result;
  }

  std::unique_ptr<ONNXRuntime> SessionCache::createSession(edm::StreamID id) const {
    // The settings of the module, plus the framework settings and the execution provider for the given stream.
    ::Ort::SessionOptions options = options_.Clone();
    edm::Service<ONNXInterface>()->configure(options, backend_, id);
    return std::make_unique<ONNXRuntime>(model_path_, &options);
  }

}  // namespace cms::Ort
