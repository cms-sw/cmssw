#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <exception>
#include <thread>
#include <random>
#include <chrono>

namespace heterogeneous {
  CPU::~CPU() noexcept(false) {}

  bool CPU::call_acquireCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    // There is no need for acquire in CPU, everything can be done in produceCPU().
    iEvent.locationSetter()(HeterogeneousDeviceId(HeterogeneousDevice::kCPU));
    waitingTaskHolder.doneWaiting(nullptr);
    return true;
  }

  void CPU::call_produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
    // For CPU we set the heterogeneous input location for produce, because there is no acquire
    // For other devices this probably doesn't make sense, because the device code is supposed to be launched from acquire.
    iEvent.setInputLocation(HeterogeneousDeviceId(HeterogeneousDevice::kCPU, 0));
    produceCPU(iEvent, iSetup);
  }

  GPUMock::GPUMock(const edm::ParameterSet& iConfig):
    enabled_(iConfig.getUntrackedParameter<bool>("GPUMock")),
    forced_(iConfig.getUntrackedParameter<std::string>("force") == "GPUMock")
  {}

  GPUMock::~GPUMock() noexcept(false) {}

  void GPUMock::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.addUntracked<bool>("GPUMock", true);
  }

  bool GPUMock::call_acquireGPUMock(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
    if(!enabled_) {
      edm::LogPrint("HeterogeneousEDProducer") << "Mock GPU is not available for this module (disabled in configuration)";
      return false;
    }

    if(!forced_) {
      // Decide randomly whether to run on GPU or CPU to simulate scheduler decisions
      std::random_device r;
      std::mt19937 gen(r());
      auto dist1 = std::uniform_int_distribution<>(0, 3); // simulate GPU (in)availability
      if(dist1(gen) == 0) {
        edm::LogPrint("HeterogeneousEDProducer") << "Mock GPU is not available (by chance)";
        return false;
      }
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
