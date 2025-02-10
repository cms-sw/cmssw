#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "HeterogeneousCore/AlpakaCore/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/HostOnlyTask.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  ESProducer::ESProducer(edm::ParameterSet const& iConfig)
      : moduleLabel_(iConfig.getParameter<std::string>("@module_label")),
        appendToDataLabel_(iConfig.getParameter<std::string>("appendToDataLabel")),
        // The 'synchronize' parameter can be unset in Alpaka modules
        // specified with the namespace prefix instead if '@alpaka'
        // suffix
        synchronize_(
            iConfig.getUntrackedParameter<edm::ParameterSet>("alpaka").getUntrackedParameter("synchronize", false)) {}

  void ESProducer::enqueueCallback(Queue& queue, edm::WaitingTaskWithArenaHolder holder) {
    edm::Service<edm::Async> async;
    auto event = cms::alpakatools::getEventCache<Event>().get(alpaka::getDev(queue));
    alpaka::enqueue(queue, *event);
    async->runAsync(
        std::move(holder),
        [event = std::move(event)]() mutable { alpaka::wait(*event); },
        []() { return "Enqueued via " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "::ESProducer::enqueueCallback()"; });
  }

  void ESProducer::throwSomeNullException() {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "The Alpaka backend has multiple devices. The device-specific produce() function returned a null product "
           "for some of the devices of the backend, but not all. This is not currently supported.";
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
