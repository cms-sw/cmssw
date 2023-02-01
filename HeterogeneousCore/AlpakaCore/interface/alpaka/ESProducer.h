#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESProducer_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/produce_helpers.h"
#include "HeterogeneousCore/AlpakaCore/interface/module_backend_config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Record.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

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
  class ESProducer : public edm::ESProducer {
    using Base = edm::ESProducer;

  public:
    static void prevalidate(edm::ConfigurationDescriptions& descriptions) {
      Base::prevalidate(descriptions);
      cms::alpakatools::module_backend_config(descriptions);
    }

  protected:
    template <typename T>
    auto setWhatProduced(T* iThis, edm::es::Label const& label = {}) {
      return setWhatProduced(iThis, &T::produce, label);
    }

    template <typename T, typename TReturn, typename TRecord>
    auto setWhatProduced(T* iThis, TReturn (T ::*iMethod)(TRecord const&), edm::es::Label const& label = {}) {
      return Base::setWhatProduced(iThis, iMethod, label);
    }

    template <typename T, typename TReturn, typename TRecord>
    auto setWhatProduced(T* iThis,
                         TReturn (T ::*iMethod)(device::Record<TRecord> const&),
                         edm::es::Label const& label = {}) {
      using TProduct = typename edm::eventsetup::produce::smart_pointer_traits<TReturn>::type;
      using ProductType = ESDeviceProduct<TProduct>;
      using ReturnType = detail::ESDeviceProductWithStorage<TProduct, TReturn>;
      return Base::setWhatProduced(
          [iThis, iMethod](TRecord const& record) -> std::unique_ptr<ProductType> {
            // TODO: move the multiple device support into EventSetup system itself
            auto const& devices = cms::alpakatools::devices<Platform>();
            std::vector<std::shared_ptr<Queue>> queues;
            queues.reserve(devices.size());
            auto ret = std::make_unique<ReturnType>(devices.size());
            bool allnull = true;
            bool anynull = false;
            for (auto const& dev : devices) {
              device::Record<TRecord> const deviceRecord(record, dev);
              auto prod = std::invoke(iMethod, iThis, deviceRecord);
              if (prod) {
                allnull = false;
                ret->insert(dev, std::move(prod));
              } else {
                anynull = true;
              }
              queues.push_back(deviceRecord.queuePtr());
            }
            // TODO: to be changed asynchronous later
            for (auto& queuePtr : queues) {
              alpaka::wait(*queuePtr);
            }
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
              ESProducer::throwSomeNullException();
            }
            return ret;
          },
          label);
    }

  private:
    static void throwSomeNullException();
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
