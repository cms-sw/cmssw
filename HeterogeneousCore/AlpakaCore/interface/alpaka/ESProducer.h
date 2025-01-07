#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESProducer_h

#include "FWCore/Framework/interface/ESProducerExternalWork.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/produce_helpers.h"
#include "HeterogeneousCore/AlpakaCore/interface/modulePrevalidate.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProductType.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Record.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include <functional>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  /**
   * The ESProducer is a base class for modules producing data into
   * the host memory space and/or the device memory space defined by
   * the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). The interface
   * looks similar to the normal edm::ESProducer.
   *
   * When producing a host product, the produce function should have
   * the the usual Record argument. For producing a device product,
   * the produce funtion should have device::Record<Record> argument.
   */
  class ESProducer : public edm::ESProducerExternalWork {
    using Base = edm::ESProducerExternalWork;

  public:
    static void prevalidate(edm::ConfigurationDescriptions& descriptions) {
      Base::prevalidate(descriptions);
      cms::alpakatools::modulePrevalidate(descriptions);
    }

  protected:
    ESProducer(edm::ParameterSet const& iConfig);

    template <typename T>
    auto setWhatProduced(T* iThis, edm::es::Label const& label = {}) {
      return setWhatProduced(iThis, &T::produce, label);
    }

    template <typename T, typename TReturn, typename TRecord>
    auto setWhatProduced(T* iThis, TReturn (T ::*iMethod)(TRecord const&), edm::es::Label const& label = {}) {
      auto cc = Base::setWhatProduced(iThis, iMethod, label);
      using TProduct = typename edm::eventsetup::produce::smart_pointer_traits<TReturn>::type;
      if constexpr (not detail::useESProductDirectly) {
        // for device backends add the copy to device
        auto tokenPtr = std::make_shared<edm::ESGetToken<TProduct, TRecord>>();
        auto ccDev = setWhatProducedDevice<TRecord>(
            [tokenPtr](device::Record<TRecord> const& iRecord) {
              using CopyT = cms::alpakatools::CopyToDevice<TProduct>;
              try {
                auto handle = iRecord.getTransientHandle(*tokenPtr);
                return std::optional{CopyT::copyAsync(iRecord.queue(), *handle)};
              } catch (edm::eventsetup::MakeDataException& e) {
                return std::optional<decltype(CopyT::copyAsync(std::declval<Queue&>(), std::declval<TProduct>()))>();
              }
            },
            label);
        *tokenPtr = ccDev.consumes(edm::ESInputTag{moduleLabel_, label.default_ + appendToDataLabel_});
      }
      return cc;
    }

    template <typename T, typename TReturn, typename TRecord>
    auto setWhatProduced(T* iThis,
                         TReturn (T ::*iMethod)(device::Record<TRecord> const&),
                         edm::es::Label const& label = {}) {
      using TProduct = typename edm::eventsetup::produce::smart_pointer_traits<TReturn>::type;
      if constexpr (detail::useESProductDirectly) {
        return Base::setWhatProduced(
            [iThis, iMethod](TRecord const& record) {
              auto const& devices = cms::alpakatools::devices<Platform>();
              assert(devices.size() == 1);
              device::Record<TRecord> const deviceRecord(record, devices.front());
              static_assert(std::is_same_v<std::remove_cvref_t<decltype(deviceRecord.queue())>,
                                           alpaka::Queue<Device, alpaka::Blocking>>,
                            "Non-blocking queue when trying to use ES data product directly. This might indicate a "
                            "need to extend the Alpaka ESProducer base class.");
              return std::invoke(iMethod, iThis, deviceRecord);
            },
            label);
      } else {
        return setWhatProducedDevice<TRecord>(
            [iThis, iMethod](device::Record<TRecord> const& record) { return std::invoke(iMethod, iThis, record); },
            label);
      }
    }

  private:
    template <typename TRecord, typename TFunc>
    auto setWhatProducedDevice(TFunc&& func, const edm::es::Label& label) {
      using Types = edm::eventsetup::impl::ReturnArgumentTypes<TFunc>;
      using TReturn = typename Types::return_type;
      using TProduct = typename edm::eventsetup::produce::smart_pointer_traits<TReturn>::type;
      using ProductType = ESDeviceProduct<TProduct>;
      using ReturnType = detail::ESDeviceProductWithStorage<TProduct, TReturn>;
      return Base::setWhatAcquiredProducedWithLambda(
          // acquire() part
          [function = std::forward<TFunc>(func), synchronize = synchronize_](TRecord const& record,
                                                                             edm::WaitingTaskWithArenaHolder holder) {
            // TODO: move the multiple device support into EventSetup system itself
            auto const& devices = cms::alpakatools::devices<Platform>();
            auto ret = std::make_unique<ReturnType>(devices.size());
            bool allnull = true;
            bool anynull = false;
            for (auto const& dev : devices) {
              device::Record<TRecord> const deviceRecord(record, dev);
              auto prod = function(deviceRecord);
              if (prod) {
                allnull = false;
                ret->insert(dev, std::move(prod));
              } else {
                anynull = true;
              }
              if (synchronize) {
                alpaka::wait(deviceRecord.queue());
              } else {
                enqueueCallback(deviceRecord.queue(), std::move(holder));
              }
              // The Queue is returned to the QueueCache. The same
              // Queue may be used for other work before the work
              // enqueued here finishes. The only impact would be a
              // (slight?) delay in the completion of the other work.
              // Given that the ESProducers are expected to be mostly
              // for host-to-device data copies, that are serialized
              // anyway (at least on current NVIDIA), this should be
              // reasonable behavior for now.
            }
            return std::tuple(std::move(ret), allnull, anynull);
          },
          // produce() part, called after the asynchronous work in all queues have finished
          [](TRecord const& record, auto fromAcquire) -> std::unique_ptr<ProductType> {
            auto [ret, allnull, anynull] = std::move(fromAcquire);
            // The 'allnull'/'anynull' actions are in produce()
            // to keep any destination memory in 'ret'
            // alive until the asynchronous work has finished
            if (allnull) {
              return nullptr;
            } else if (anynull) {
              // TODO: throwing an exception if the iMethod() returns
              // null for some of th devices of one backend is
              // suboptimal. On the other hand, in the near term
              // multiple devices per backend is useful only for
              // private tests (not production), and the plan is to
              // make the EventSetup system itself aware of multiple
              // devies (or memory spaces). I hope this exception
              // would be good-enough until we get there.
              throwSomeNullException();
            }
            return std::move(ret);
          },
          label);
    }

    static void enqueueCallback(Queue& queue, edm::WaitingTaskWithArenaHolder holder);
    static void throwSomeNullException();

    std::string const moduleLabel_;
    std::string const appendToDataLabel_;
    bool const synchronize_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
