#ifndef HeterogeneousCore_AcceleratorService_AcceleratorService_h
#define HeterogeneousCore_AcceleratorService_AcceleratorService_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include <memory>
#include <mutex>
#include <vector>

namespace edm {
  class Event;
  class ParameterSet;
  class ActivityRegistry;
  class ModuleDescription;
  namespace service {
    class SystemBounds;
  }
}

namespace accelerator {
  // Inheritance vs. type erasure? I'm now going with the latter even
  // if it is more complex to setup and maintain in order to support a
  // case where a single class implements multiple CPU/GPU/etc
  // interfaces, in which case via inheritance we can't separate the
  // cases in the scheduling interface.
  //
  // Want a base class in order to have the per-device calls to be
  // made non-inlined (how necessary is this?)
  //
  // Note that virtual destructors are not needed in the base classes
  // as the pattern is to construct the concrete class in stack and
  // keep that object around as long as the function call accessing
  // the object via base class pointer/reference is finished.

  /**
   * CPU algorithm
   *
   * The T can be any class implementing "runCPU(void)" method (return
   * value is ignored)
   *
   * A CPU algorithm is run synchronously in the same TBB task, i.e.
   * when everything is finished when runCPU() call returns.
   */
  class AlgoCPUBase {
  public:
    AlgoCPUBase() {}
    virtual void runCPU() = 0;
  };
  template <typename T> class AlgoCPU: public AlgoCPUBase {
  public:
    AlgoCPU(T *algo): algo_(algo) {}
    void runCPU() override { algo_->runCPU(); }
  private:
    T *algo_;
  };
  template <typename T> AlgoCPU<T> algoCPU(T *algo) { return AlgoCPU<T>(algo); }

  /**
   * GPU mock algorithm
   *
   * The T can be any class implementing
   * "runGPUMock(std::function<void()>)" method (return value is
   * ignored).
   *
   * The implemented method must call the callback function when all
   * work is finished. If any of the work is asynchronous, it is up to
   * the algorithm to launch the work asynchronously and ensure that
   * the callback is called after all asynchronous work is finished.
   */
  class AlgoGPUMockBase {
  public:
    AlgoGPUMockBase() {}
    virtual void runGPUMock(std::function<void()> callback) = 0;
  };
  template <typename T> class AlgoGPUMock: public AlgoGPUMockBase {
  public:
    AlgoGPUMock(T *algo): algo_(algo) {}
    void runGPUMock(std::function<void()> callback) override { algo_->runGPUMock(std::move(callback)); }
  private:
    T *algo_;
  };
  template <typename T> AlgoGPUMock<T> algoGPUMock(T *algo) { return AlgoGPUMock<T>(algo); }

  /**
   * GPU CUDA algorithm
   *
   * The T can be any class implementing
   * "runGPUCuda(std::function<void()>)" method (return value is
   * ignored).
   *
   * The implemented method must call the callback function when all
   * work is finished. If any of the work is asynchronous, it is up to
   * the algorithm to launch the work asynchronously and ensure that
   * the callback is called after all asynchronous work is finished.
   *
   * For CUDA the above conditions mean using
   * - CUDA stream
   * - asynchronous memory transfers
   * - asynchronous kernel launches
   * - registering the callback to the stream after all other work has been launched
   *
   * Note that the CUDA stream object must live at least as long as
   * the callback() is called (in practice a tiny bit later).
   */
  class AlgoGPUCudaBase {
  public:
    AlgoGPUCudaBase() {}
    virtual void runGPUCuda(std::function<void()> callback) = 0;
  };
  template <typename T> class AlgoGPUCuda: public AlgoGPUCudaBase {
  public:
    AlgoGPUCuda(T *algo): algo_(algo) {}
    void runGPUCuda(std::function<void()> callback) override { algo_->runGPUCuda(std::move(callback)); }
  private:
    T *algo_;
  };
  template <typename T> AlgoGPUCuda<T> algoGPUCuda(T *algo) { return AlgoGPUCuda<T>(algo); }
}

/**
 * Prototype for a service for scheduling heterogeneous algorithms
 *
 * It can be that in the end we don't need a Service, but for now the
 * protyping proceeds with one.
 *
 * At the moment the client EDModules must use the ExternalWork extension.
 *
 * Client EDModules must first register themselves by calling the
 * book() method in their constructor and storing the Token object as
 * member variables.
 *
 * In the aqcuire(), the clients must call the schedule() method to
 * schedule (and possibly run) the algorithms (for more details see
 * the documentation of the method).
 *
 * In the produce(), the clients must check with
 * algoExecutionLocation() which algorithm was run, fetch the output
 * of that algorithm, and put it in the Event.
 */
class AcceleratorService {
public:
  class Token {
  public:
    explicit Token(unsigned int id): id_(id) {}

    unsigned int id() const { return id_; }
  private:
    unsigned int id_;
  };

  AcceleratorService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);

  Token book(); // TODO: better name, unfortunately 'register' is a reserved keyword...

  /**
   * Schedule the various versions of the algorithm to the available
   * heterogeneous devices.
   *
   * The parameter pack is an ordered list of accelerator::Algo*<T>
   * objects (note the helper functions to create them). The order of
   * the algorithms is taken as the preferred order to be tried. I.e.
   * the code tries to schedule the first algorithm, if that fails (no
   * device, to be extended) try the next one etc. The CPU version has
   * to be the last one.
   *
   *
   * TODO: passing the "input" parameter here is a bit boring, but
   * somehow we have to schedule according to the input. Try to think
   * something better.
   */
  template <typename I, typename... Args>
  void schedule(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input, Args&&... args) {
    scheduleImpl(token, streamID, std::move(waitingTaskHolder), input, std::forward<Args>(args)...);
  }
  HeterogeneousDeviceId algoExecutionLocation(Token token, edm::StreamID streamID) const {
    return algoExecutionLocation_[tokenStreamIdsToDataIndex(token.id(), streamID)];
  }

private:
  // signals
  void preallocate(edm::service::SystemBounds const& bounds);
  void preModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleConstruction(edm::ModuleDescription const& desc);


  // other helpers
  unsigned int tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const;

  // experimenting new interface
  template <typename I, typename A, typename... Args>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input,
                    accelerator::AlgoGPUMock<A> gpuMockAlgo, Args&&... args) {
    bool succeeded = true;
    if(input) {
      succeeded = input->isProductOn(HeterogeneousDevice::kGPUMock);
    }
    if(succeeded) {
      succeeded = scheduleGPUMock(token, streamID, waitingTaskHolder, gpuMockAlgo);
    }
    if(!succeeded) {
      scheduleImpl(token, streamID, std::move(waitingTaskHolder), input, std::forward<Args>(args)...);
    }
  }
  template <typename I, typename A, typename... Args>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input,
                    accelerator::AlgoGPUCuda<A> gpuCudaAlgo, Args&&... args) {
    bool succeeded = true;
    if(input) {
      succeeded = input->isProductOn(HeterogeneousDevice::kGPUCuda);
    }
    if(succeeded) {
      succeeded = scheduleGPUCuda(token, streamID, waitingTaskHolder, gpuCudaAlgo);
    }
    if(!succeeded)
      scheduleImpl(token, streamID, std::move(waitingTaskHolder), input, std::forward<Args>(args)...);
  }
  // Break recursion, require CPU to be the last
  template <typename I, typename A>
  void scheduleImpl(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, const I *input,
                    accelerator::AlgoCPU<A> cpuAlgo) {
    scheduleCPU(token, streamID, std::move(waitingTaskHolder), cpuAlgo);
  }
  bool scheduleGPUMock(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUMockBase& gpuMockAlgo);
  bool scheduleGPUCuda(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUCudaBase& gpuCudaAlgo);
  void scheduleCPU(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoCPUBase& cpuAlgo);

  
  unsigned int numberOfStreams_ = 0;

  // nearly (if not all) happens multi-threaded, so we need some
  // thread-locals to keep track in which module we are
  static thread_local unsigned int currentModuleId_;
  static thread_local std::string currentModuleLabel_; // only for printouts

  // TODO: how to treat subprocesses?
  std::mutex moduleMutex_;
  std::vector<unsigned int> moduleIds_;                      // list of module ids that have registered something on the service
  std::vector<HeterogeneousDeviceId> algoExecutionLocation_;
};

#endif
