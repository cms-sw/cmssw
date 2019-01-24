#ifndef HeterogeneousCore_Producer_HeterogeneousEvent_h
#define HeterogeneousCore_Producer_HeterogeneousEvent_h

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousDeviceId.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

namespace edm {
  class HeterogeneousEvent {
  public:
    HeterogeneousEvent(const edm::Event *event, HeterogeneousDeviceId *location): constEvent_(event), location_(location) {}
    HeterogeneousEvent(edm::Event *event, HeterogeneousDeviceId *location): event_(event), constEvent_(event), location_(location) {}

    // Accessors to members
    edm::Event& event() { return *event_; }
    const edm::Event& event() const { return *constEvent_; }

    // For the "acquire" phase, the "input location" is used for
    // scheduling, while "location" is used to set the location where
    // the algorithm was run
    void setInputLocation(HeterogeneousDeviceId location) { inputLocation_ = location; }

    std::function<void(HeterogeneousDeviceId)> locationSetter() {
      return [loc=location_](HeterogeneousDeviceId location) { *loc = location; };
    }
    const HeterogeneousDeviceId& location() const { return *location_; }

    // Delegate to edm::Event
    auto id() const { return constEvent_->id(); }
    auto streamID() const { return constEvent_->streamID(); }


    template <typename Product, typename Token, typename Type>
    bool getByToken(const Token& token, edm::Handle<Type>& handle) const {
      edm::Handle<HeterogeneousProduct> tmp;
      constEvent_->getByToken(token, tmp);
      if(tmp.failedToGet()) {
        auto copy = tmp.whyFailedFactory();
        handle = edm::Handle<Type>(std::move(copy));
        return false;
      }
      if(tmp.isValid()) {
#define CASE(ENUM) case ENUM: this->template get<ENUM, Product>(handle, tmp, 0); break
        switch(inputLocation_.deviceType()) {
        CASE(HeterogeneousDevice::kCPU);
        CASE(HeterogeneousDevice::kGPUMock);
        CASE(HeterogeneousDevice::kGPUCuda);
        default:
          throw cms::Exception("LogicError") << "edm::HeterogeneousEvent::getByToken(): no case statement for device " << static_cast<unsigned int>(location().deviceType()) << ". If you are calling getByToken() from produceX() where X != CPU, please move the call to acquireX().";
        }
#undef CASE
        return true;
      }
      return false;
    }

    // Delegate standard getByToken to edm::Event
    template <typename Token, typename Type>
    bool getByToken(const Token& token, edm::Handle<Type>& handle) const {
      return constEvent_->getByToken(token, handle);
    }

    template <typename PROD>
    auto put(std::unique_ptr<PROD> product) {
      return event_->put(std::move(product));
    }

    template <typename PROD>
    auto put(std::unique_ptr<PROD> product, std::string const& productInstanceName) {
      return event_->put(std::move(product), productInstanceName);
    }

    template <typename Product, typename Type>
    void put(std::unique_ptr<Type> product) {
      assert(location().deviceType() == HeterogeneousDevice::kCPU);
      event_->put(std::make_unique<HeterogeneousProduct>(Product(heterogeneous::HeterogeneousDeviceTag<HeterogeneousDevice::kCPU>(), std::move(*product))));
    }

    template <typename Product, typename Type, typename F>
    auto put(std::unique_ptr<Type> product, F transferToCPU) {
      std::unique_ptr<HeterogeneousProduct> prod;
#define CASE(ENUM) case ENUM: this->template make<ENUM, Product>(prod, std::move(product), std::move(transferToCPU), 0); break
      switch(location().deviceType()) {
      CASE(HeterogeneousDevice::kGPUMock);
      CASE(HeterogeneousDevice::kGPUCuda);
      default:
        throw cms::Exception("LogicError") << "edm::HeterogeneousEvent::put(): no case statement for device " << static_cast<unsigned int>(location().deviceType());
      }
#undef CASE
      return event_->put(std::move(prod));
    }

  private:
    template <HeterogeneousDevice Device, typename Product, typename Type>
    typename std::enable_if_t<Product::template CanGet<Device, Type>::value, void>
    get(edm::Handle<Type>& dst, const edm::Handle<HeterogeneousProduct>& src, int) const {
      const auto& concrete = src->get<Product>();
      const auto& provenance = src.provenance();
      dst = edm::Handle<Type>(&(concrete.template getProduct<Device>()), provenance);
    }
    template <HeterogeneousDevice Device, typename Product, typename Type>
    void get(edm::Handle<Type>& dst, const edm::Handle<HeterogeneousProduct>& src, long) const {
      throw cms::Exception("Assert") << "Invalid call to get, Device " << static_cast<int>(Device)
                                     << " Product " << typeid(Product).name()
                                     << " Type " << typeid(Type).name()
                                     << " CanGet::FromType " << typeid(typename Product::template CanGet<Device, Type>::FromType).name()
                                     << " CanGet::value " << Product::template CanGet<Device, Type>::value;
    }

    template<HeterogeneousDevice Device, typename Product, typename Type, typename F>
    typename std::enable_if_t<Product::template CanPut<Device, Type>::value, void>
    make(std::unique_ptr<HeterogeneousProduct>& ret, std::unique_ptr<Type> product, F transferToCPU, int) {
      ret = std::make_unique<HeterogeneousProduct>(Product(heterogeneous::HeterogeneousDeviceTag<Device>(),
                                                           std::move(*product), location(), std::move(transferToCPU)));
    }
    template<HeterogeneousDevice Device, typename Product, typename Type, typename F>
    void make(std::unique_ptr<HeterogeneousProduct>& ret, std::unique_ptr<Type> product, F transferToCPU, long) {
      throw cms::Exception("Assert") << "Invalid call to make, Device " << static_cast<int>(Device)
                                     << " Product " << typeid(Product).name()
                                     << " Type " << typeid(Type).name()
                                     << " CanPut::ToType " << typeid(typename Product::template CanPut<Device, Type>::ToType).name()
                                     << " CanPut::value " << Product::template CanPut<Device, Type>::value;
    }

    edm::Event *event_ = nullptr;
    const edm::Event *constEvent_ = nullptr;
    HeterogeneousDeviceId inputLocation_;
    HeterogeneousDeviceId *location_ = nullptr;
  };
} // end namespace edm

#endif
