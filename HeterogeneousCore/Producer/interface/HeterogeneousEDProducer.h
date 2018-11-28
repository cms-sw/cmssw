#ifndef HeterogeneousCore_Producer_HeterogeneousEDProducer_h
#define HeterogeneousCore_Producer_HeterogeneousEDProducer_h

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"
#include "HeterogeneousCore/Producer/interface/DeviceWrapper.h"

namespace heterogeneous {
  class CPU {
  public:
    explicit CPU(const edm::ParameterSet& iConfig) {}
    virtual ~CPU() noexcept(false);

    static void fillPSetDescription(edm::ParameterSetDescription desc) {}

    void call_beginStreamCPU(edm::StreamID id) {
      beginStreamCPU(id);
    }
    bool call_acquireCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup);

  private:
    virtual void beginStreamCPU(edm::StreamID id) {};
    virtual void produceCPU(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) = 0;
  };
  DEFINE_DEVICE_WRAPPER(CPU, HeterogeneousDevice::kCPU);

  class GPUMock {
  public:
    explicit GPUMock(const edm::ParameterSet& iConfig);
    virtual ~GPUMock() noexcept(false);

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

    void call_beginStreamGPUMock(edm::StreamID id) {
      beginStreamGPUMock(id);
    }
    bool call_acquireGPUMock(DeviceBitSet inputLocation, edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder);
    void call_produceGPUMock(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
      produceGPUMock(iEvent, iSetup);
    }

  private:
    virtual void beginStreamGPUMock(edm::StreamID id) {};
    virtual void acquireGPUMock(const edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup, std::function<void()> callback) = 0;
    virtual void produceGPUMock(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) = 0;

    const bool enabled_;
    const bool forced_;
  };
  DEFINE_DEVICE_WRAPPER(GPUMock, HeterogeneousDevice::kGPUMock);
}

namespace heterogeneous {
  ////////////////////
  template <typename ...Args>
  struct CallBeginStream;
  template <typename T, typename D, typename ...Devices>
  struct CallBeginStream<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, Args&&... args) {
      // may not perfect-forward here in order to be able to forward arguments to next CallBeginStream.
      Mapping<D>::beginStream(ref, args...);
      CallBeginStream<T, Devices...>::call(ref, std::forward<Args>(args)...);
    }
  };
  // break recursion and require CPU to be the last
  template <typename T>
  struct CallBeginStream<T, CPU> {
    template <typename ...Args>
    static void call(T& ref, Args&&... args) {
      Mapping<CPU>::beginStream(ref, std::forward<Args>(args)...);
    }
  };

  ////////////////////
  template <typename ...Args>
  struct CallAcquire;
  template <typename T, typename D, typename ...Devices>
  struct CallAcquire<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousProductBase *input, Args&&... args) {
      bool succeeded = true;
      DeviceBitSet inputLocation;
      if(input) {
        succeeded = input->isProductOn(Mapping<D>::deviceEnum);
        if(succeeded) {
          inputLocation = input->onDevices(Mapping<D>::deviceEnum);
        }
      }
      if(succeeded) {
        // may not perfect-forward here in order to be able to forward arguments to next CallAcquire.
        succeeded = Mapping<D>::acquire(ref, inputLocation, args...);
      }
      if(!succeeded) {
        CallAcquire<T, Devices...>::call(ref, input, std::forward<Args>(args)...);
      }
    }
  };
  // break recursion and require CPU to be the last
  template <typename T>
  struct CallAcquire<T, CPU> {
    template <typename ...Args>
    static void call(T& ref, const HeterogeneousProductBase *input, Args&&... args) {
      Mapping<CPU>::acquire(ref, std::forward<Args>(args)...);
    }
  };

  ////////////////////
  template <typename ...Args>
  struct CallProduce;
  template <typename T, typename D, typename ...Devices>
  struct CallProduce<T, D, Devices...> {
    template <typename ...Args>
    static void call(T& ref, edm::HeterogeneousEvent& iEvent, Args&&... args) {
      if(iEvent.location().deviceType() == Mapping<D>::deviceEnum) {
        Mapping<D>::produce(ref, iEvent, std::forward<Args>(args)...);
      }
      else {
        CallProduce<T, Devices...>::call(ref, iEvent, std::forward<Args>(args)...);
      }
    }
  };
  template <typename T>
  struct CallProduce<T> {
    template <typename ...Args>
    static void call(T& ref, Args&&... args) {}
  };


  template <typename ...Devices>
  class HeterogeneousDevices: public Devices... {
  public:
    explicit HeterogeneousDevices(const edm::ParameterSet& iConfig): Devices(iConfig)... {}

    static void fillPSetDescription(edm::ParameterSetDescription& desc) {
      // The usual trick to expand the parameter pack for function call
      using expander = int[];
      (void)expander {0, ((void)Devices::fillPSetDescription(desc), 1)... };
      desc.addUntracked<std::string>("force", "");
    }

    void call_beginStream(edm::StreamID id) {
      CallBeginStream<HeterogeneousDevices, Devices...>::call(*this, id);
    }

    void call_acquire(const HeterogeneousProductBase *input,
                      edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup,
                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
      CallAcquire<HeterogeneousDevices, Devices...>::call(*this, input, iEvent, iSetup, std::move(waitingTaskHolder));
    }

    void call_produce(edm::HeterogeneousEvent& iEvent, const edm::EventSetup& iSetup) {
      CallProduce<HeterogeneousDevices, Devices...>::call(*this, iEvent, iSetup);
    }
  };
} // end namespace heterogeneous


template <typename Devices, typename ...Capabilities>
class HeterogeneousEDProducer: public Devices, public edm::stream::EDProducer<edm::ExternalWork, Capabilities...> {
public:
  explicit HeterogeneousEDProducer(const edm::ParameterSet& iConfig):
    Devices(iConfig.getUntrackedParameter<edm::ParameterSet>("heterogeneousEnabled_"))
  {}
  ~HeterogeneousEDProducer() override = default;

protected:
  edm::EDGetTokenT<HeterogeneousProduct> consumesHeterogeneous(const edm::InputTag& tag) {
    tokens_.push_back(this->template consumes<HeterogeneousProduct>(tag));
    return tokens_.back();
  }

  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    edm::ParameterSetDescription nested;
    Devices::fillPSetDescription(nested);
    desc.addUntracked<edm::ParameterSetDescription>("heterogeneousEnabled_", nested);
  }

private:
  void beginStream(edm::StreamID id) override {
    Devices::call_beginStream(id);
  }

  void acquire(const edm::Event& iEvent, const edm::EventSetup& iSetup, edm::WaitingTaskWithArenaHolder waitingTaskHolder) final {
    const HeterogeneousProductBase *input = nullptr;

    std::vector<const HeterogeneousProduct *> products;
    for(const auto& token: tokens_) {
      edm::Handle<HeterogeneousProduct> handle;
      iEvent.getByToken(token, handle);
      if(handle.isValid()) {
        // let the user acquire() code to deal with missing products
        // (and hope they don't mess up the scheduling!)
        products.push_back(handle.product());
      }
    }
    if(!products.empty()) {
      // TODO: check all inputs, not just the first one
      input = products[0]->getBase();
    }

    auto eventWrapper = edm::HeterogeneousEvent(&iEvent, &algoExecutionLocation_);
    Devices::call_acquire(input, eventWrapper, iSetup, std::move(waitingTaskHolder));
  }

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) final {
    if(algoExecutionLocation_.deviceType() == HeterogeneousDeviceId::kInvalidDevice) {
      // TODO: eventually fall back to CPU
      throw cms::Exception("LogicError") << "Trying to produce(), but algorithm was not executed successfully anywhere?";
    }
    auto eventWrapper = edm::HeterogeneousEvent(&iEvent, &algoExecutionLocation_);
    Devices::call_produce(eventWrapper, iSetup);
  }

  std::vector<edm::EDGetTokenT<HeterogeneousProduct> > tokens_;
  HeterogeneousDeviceId algoExecutionLocation_;
};

#endif



