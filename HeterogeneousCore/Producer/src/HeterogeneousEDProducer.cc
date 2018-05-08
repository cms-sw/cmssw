#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <exception>
#include <thread>
#include <random>
#include <chrono>

namespace heterogeneous {
  CPU::~CPU() noexcept(false) {}

  bool CPU::call_acquireCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    std::exception_ptr exc;
    try {
      iEvent.setInputLocation(HeterogeneousDeviceId(HeterogeneousDevice::kCPU));
      acquireCPU(iEvent, iSetup);
      iEvent.locationSetter()(HeterogeneousDeviceId(HeterogeneousDevice::kCPU));
    } catch(...) {
      exc = std::current_exception();
    }
    waitingTaskHolder.doneWaiting(exc);
    return true;
  }

  GPUMock::~GPUMock() noexcept(false) {}

  bool GPUMock::call_acquireGPUMock(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    // Decide randomly whether to run on GPU or CPU to simulate scheduler decisions
    std::random_device r;
    std::mt19937 gen(r());
    auto dist1 = std::uniform_int_distribution<>(0, 3); // simulate GPU (in)availability
    if(dist1(gen) == 0) {
      edm::LogPrint("HeterogeneousEDProducer") << "Mock GPU is not available (by chance)";
      return false;
    }

    try {
      iEvent.setInputLocation(HeterogeneousDeviceId(HeterogeneousDevice::kGPUMock, 0));
      acquireGPUMock(iEvent, iSetup,
                     [waitingTaskHolder, // copy needed for the catch block
                      locationSetter=iEvent.locationSetter(),
                      location=&(iEvent.location())
                      ]() mutable {
                       locationSetter(HeterogeneousDeviceId(HeterogeneousDevice::kGPUMock, 0));
                       waitingTaskHolder.doneWaiting(nullptr);
                     });
    } catch(...) {
      waitingTaskHolder.doneWaiting(std::current_exception());
    }
    return true;
  }
}
