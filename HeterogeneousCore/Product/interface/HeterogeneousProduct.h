#ifndef HeterogeneousCore_Product_HeterogeneousProduct_h
#define HeterogeneousCore_Product_HeterogeneousProduct_h

#include "FWCore/Utilities/interface/Exception.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousDeviceId.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProductBase.h"

#include <cassert>
#include <functional>
#include <memory>
#include <tuple>

namespace heterogeneous {
  template <typename T>
  std::string bitsetArrayToString(const T& bitsetArray) {
    std::string ret;
    for(const auto& bitset: bitsetArray) {
      ret += bitset.to_string() + " ";
    }
    return ret;
  }

  /**
   * The *Product<T> templates are to specify in a generic way which
   * data locations and device-specific types the
   * HeterogeneousProduct<> supports.
   *
   * Helper functions are provided to infer the type from input
   * arguments to ease the construction of HeterogeneousProduct<>
   *
   * TODO: try to simplify...
   */

  // Mapping from *Product<T> to HeterogeneousDevice enumerator
  template <typename T> struct ProductToEnum {};

#define DEFINE_DEVICE_PRODUCT(ENUM) \
    template <typename T> \
    class ENUM##Product { \
    public: \
      using DataType = T; \
      static constexpr const HeterogeneousDevice tag = HeterogeneousDevice::k##ENUM; \
      ENUM##Product() = default; \
      ENUM##Product(T&& data): data_(std::move(data)) {} \
      const T& product() const { return data_; } \
      T& product() { return data_; } \
    private: \
      T data_; \
    }; \
    template <typename T> struct ProductToEnum<ENUM##Product<T>> { static constexpr const HeterogeneousDevice value = HeterogeneousDevice::k##ENUM; }

  DEFINE_DEVICE_PRODUCT(CPU);
  DEFINE_DEVICE_PRODUCT(GPUMock);
  DEFINE_DEVICE_PRODUCT(GPUCuda);
#undef DEFINE_DEVICE_PRODUCT

  // Tag class to allow disabling automatic device->CPU transfers
  struct DisableTransfer {};

  /**
   * Below are various helpers
   *
   * TODO: move to an inner namespace (e.g. detail, impl)?, possibly to a separate file
   */
  
  // Empty struct for tuple defitionons
  struct Empty {};

  // Metaprogram to return the *Product<T> type for a given enumerator if it exists in Types... pack
  template <HeterogeneousDevice device, typename... Types>
  struct IfInPack;

  template <HeterogeneousDevice device, typename Type, typename... Types>
  struct IfInPack<device, Type, Types...> {
    using type = std::conditional_t<device==Type::tag,
                                    Type,
                                    typename IfInPack<device, Types...>::type >;
  };
  template <HeterogeneousDevice device>
  struct IfInPack<device> {
    using type = Empty;
  };

  template <HeterogeneousDevice device, typename... Types>
  using IfInPack_t = typename IfInPack<device, Types...>::type;

  // Metaprogram to construct the callback function type for device->CPU transfers
  template <typename CPUProduct, typename DeviceProduct>
  struct CallBackType {
    using type = std::function<void(typename DeviceProduct::DataType const&, typename CPUProduct::DataType&)>;
  };
  template <typename CPUProduct>
  struct CallBackType<CPUProduct, Empty> {
    using type = Empty;
  };
  template <typename CPUProduct, typename DeviceProductOrEmpty>
  using CallBackType_t = typename CallBackType<CPUProduct, DeviceProductOrEmpty>::type;

  // Metaprogram to get an element from a tuple, or Empty if out of bounds
  template <size_t index, typename Tuple, typename Enable=void>
  struct TupleElement {
    using type = Empty;
  };
  template <size_t index, typename Tuple>
  struct TupleElement<index, Tuple, typename std::enable_if<(index < std::tuple_size<Tuple>::value)>::type> {
    using type = std::tuple_element_t<index, Tuple>;
  };
  template <size_t index, typename Tuple>
  using TupleElement_t = typename TupleElement<index, Tuple>::type;


  // Metaprogram to loop over two tuples and an array of bitsets (of
  // equal length), and if any element of bitset is set to true call a
  // function from one of the tuples with arguments from the second
  // tuple
  template <typename FunctionTuple, typename ProductTuple, typename BitSetArray, typename FunctionTupleElement, size_t sizeMinusIndex>
  struct CallFunctionIf {
    static bool call(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSetArray& bitsetArray) {
      constexpr const auto index = std::tuple_size<BitSetArray>::value - sizeMinusIndex;
      if(bitsetArray[index].any()) {
        const auto& func = std::get<index>(functionTuple);
        if(!func) {
          throw cms::Exception("Assert") << "Attempted to call transfer-to-CPU function for device " << index << " but the std::function object is not valid!";
        }
        func(std::get<index>(productTuple).product(), std::get<0>(productTuple).product());
        return true;
      }
      return CallFunctionIf<FunctionTuple, ProductTuple, BitSetArray,
                            TupleElement_t<index+1, FunctionTuple>, sizeMinusIndex-1>::call(functionTuple, productTuple, bitsetArray);
    }
  };
  template <typename FunctionTuple, typename ProductTuple, typename BitSetArray, size_t sizeMinusIndex>
  struct CallFunctionIf<FunctionTuple, ProductTuple, BitSetArray, Empty, sizeMinusIndex> {
    static bool call(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSetArray& bitsetArray) {
      constexpr const auto index = std::tuple_size<BitSetArray>::value - sizeMinusIndex;
      return CallFunctionIf<FunctionTuple, ProductTuple, BitSetArray,
                            TupleElement_t<index+1, FunctionTuple>, sizeMinusIndex-1>::call(functionTuple, productTuple, bitsetArray);
    }
  };
  template <typename FunctionTuple, typename ProductTuple, typename BitSetArray>
  struct CallFunctionIf<FunctionTuple, ProductTuple, BitSetArray, Empty, 0> {
    static bool call(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSetArray& bitsetArray) {
      return false;
    }
  };

  // Metaprogram to specialize getProduct() for CPU
  template <HeterogeneousDevice device>
  struct GetOrTransferProduct {
    template <typename FunctionTuple, typename ProductTuple, typename BitSetArray>
    static const auto& getProduct(const FunctionTuple& functionTuple, ProductTuple& productTuple, const BitSetArray& location) {
      constexpr const auto index = static_cast<unsigned int>(device);
      if(!location[index].any()) {
        throw cms::Exception("LogicError") << "Called getProduct() for device " << index << " but the data is not there! Location bitfield is " << bitsetArrayToString(location);
      }
      return std::get<index>(productTuple).product();
    }
  };

  template <>
  struct GetOrTransferProduct<HeterogeneousDevice::kCPU> {
    template <typename FunctionTuple, typename ProductTuple, typename BitSetArray>
    static const auto& getProduct(const FunctionTuple& functionTuple, ProductTuple& productTuple, BitSetArray& location) {
      constexpr const auto index = static_cast<unsigned int>(HeterogeneousDevice::kCPU);
      if(!location[index].any()) {
        auto found = CallFunctionIf<FunctionTuple, ProductTuple, BitSetArray,
                                    std::tuple_element_t<1, FunctionTuple>, std::tuple_size<BitSetArray>::value-1>::call(functionTuple, productTuple, location);
        if(!found) {
          throw cms::Exception("LogicError") << "Attempted to transfer data to CPU, but the data is not available anywhere! Location bitfield is " << bitsetArrayToString(location);
        }
      }
      location[index].set(0);
      return std::get<index>(productTuple).product();
    }
  };

  // Metaprogram to return DataType or Empty
  template <typename T>
  struct DataTypeOrEmpty {
    using type = typename T::DataType;
  };
  template<>
  struct DataTypeOrEmpty<Empty> {
    using type = Empty;
  };
}

/**
 * Generic data product for holding data on CPU or a heterogeneous
 * device which keeps track where the data is. Data can be
 * automatically transferred from the device to CPU when data is
 * requested on CPU but does not exist there (yet).
 *
 * TODO:
 * * extend transfers to device->device (within a single device type)
 */
template <typename CPUProduct, typename... Types>
class HeterogeneousProductImpl: public HeterogeneousProductBase {
  using ProductTuple = std::tuple<CPUProduct,
                                  heterogeneous::IfInPack_t<HeterogeneousDevice::kGPUMock, Types...>,
                                  heterogeneous::IfInPack_t<HeterogeneousDevice::kGPUCuda, Types...>
                                  >;
  using TransferToCPUTuple = std::tuple<heterogeneous::Empty, // no need to transfer from CPU to CPU
                                        heterogeneous::CallBackType_t<CPUProduct, std::tuple_element_t<static_cast<unsigned int>(HeterogeneousDevice::kGPUMock), ProductTuple>>,
                                        heterogeneous::CallBackType_t<CPUProduct, std::tuple_element_t<static_cast<unsigned int>(HeterogeneousDevice::kGPUCuda), ProductTuple>>
                                        >;
  // Some sanity checks
  static_assert(std::tuple_size<ProductTuple>::value == std::tuple_size<TransferToCPUTuple>::value, "Size mismatch");
  static_assert(std::tuple_size<ProductTuple>::value == static_cast<unsigned int>(HeterogeneousDevice::kSize), "Size mismatch");
public:
  template <HeterogeneousDevice Device, typename Type>
  struct CanGet {
    using FromType = typename heterogeneous::DataTypeOrEmpty<std::tuple_element_t<static_cast<unsigned int>(Device), ProductTuple> >::type;
    static const bool value = std::is_same<Type, FromType>::value;
  };

  template<HeterogeneousDevice Device, typename Type>
  struct CanPut {
    using ToType = typename heterogeneous::DataTypeOrEmpty<std::tuple_element_t<static_cast<unsigned int>(Device), ProductTuple> >::type;
    static const bool value = std::is_same<ToType, Type>::value;
  };

  HeterogeneousProductImpl() = default;
  ~HeterogeneousProductImpl() override = default;
  HeterogeneousProductImpl(HeterogeneousProductImpl<CPUProduct, Types...>&& other) {
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lk1(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lk2(other.mutex_, std::adopt_lock);

    products_ = std::move(other.products_);
    transfersToCPU_ = std::move(other.transfersToCPU_);
    location_ = std::move(other.location_);
  }
  HeterogeneousProductImpl<CPUProduct, Types...>& operator=(HeterogeneousProductImpl<CPUProduct, Types...>&& other) {
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lk1(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lk2(other.mutex_, std::adopt_lock);

    products_ = std::move(other.products_);
    transfersToCPU_ = std::move(other.transfersToCPU_);
    location_ = std::move(other.location_);
    return *this;
  }

  // Constructor for CPU data
  template <HeterogeneousDevice Device, typename D>
  HeterogeneousProductImpl(heterogeneous::HeterogeneousDeviceTag<Device>, D&& data) {
    static_assert(Device == HeterogeneousDevice::kCPU, "This overload allows only CPU device");
    constexpr const auto index = static_cast<unsigned int>(HeterogeneousDevice::kCPU);
    std::get<index>(products_) = std::move(data);
    location_[index].set(0);
  }

  /**
   * Generic constructor for device data. A function to transfer the
   * data to CPU has to be provided as well.
   */
  template <HeterogeneousDevice Device, typename D, typename F>
  HeterogeneousProductImpl(heterogeneous::HeterogeneousDeviceTag<Device>, D&& data, HeterogeneousDeviceId location, F transferToCPU) {
    constexpr const auto index = static_cast<unsigned int>(Device);
    assert(location.deviceType() == Device);
    std::get<index>(products_) = std::move(data);
    std::get<index>(transfersToCPU_) = std::move(transferToCPU);
    location_[index].set(location.deviceId());
  }

  /**
   * Generic constructor for device data, but without the transfer function(!).
   */
  template <HeterogeneousDevice Device, typename D>
  HeterogeneousProductImpl(heterogeneous::HeterogeneousDeviceTag<Device>, D&& data, HeterogeneousDeviceId location, heterogeneous::DisableTransfer) {
    // TODO: try to avoid code duplication between the other device data
    constexpr const auto index = static_cast<unsigned int>(Device);
    assert(location.deviceType() == Device);
    std::get<index>(products_) = std::move(data);
    location_[index].set(location.deviceId());
  }

  template <HeterogeneousDevice device>
  const auto& getProduct() const {
    constexpr const auto index = static_cast<unsigned int>(device);
    static_assert(!std::is_same<std::tuple_element_t<index, ProductTuple>,
                                heterogeneous::Empty>::value,
                  "This HeterogeneousProduct does not support this type");

    // Locking the mutex here is quite "conservative"
    // Writes happen only if the "device" is CPU and the data is elsewhere
    std::lock_guard<std::mutex> lk(mutex_);
    return heterogeneous::GetOrTransferProduct<device>::getProduct(transfersToCPU_, products_, location_);
  }

private:
  mutable ProductTuple products_;
  TransferToCPUTuple transfersToCPU_;
};

/**
 * The main purpose of the HeterogeneousProduct,
 * HeterogeneousProductBase, HeterogeneousProductImpl<...> class
 * hierarchy is to avoid the dictionary generation for the concrete
 * HeterogeneousProductImpl<...>'s.
 */
class HeterogeneousProduct {
public:
  HeterogeneousProduct() = default;

  template <typename... Args>
  HeterogeneousProduct(HeterogeneousProductImpl<Args...>&& impl) {
    impl_.reset(static_cast<HeterogeneousProductBase *>(new HeterogeneousProductImpl<Args...>(std::move(impl))));
  }

  HeterogeneousProduct(HeterogeneousProduct&&) = default;
  HeterogeneousProduct& operator=(HeterogeneousProduct&&) = default;

  ~HeterogeneousProduct() = default;

  bool isNonnull() const { return static_cast<bool>(impl_); }
  bool isNull() const { return !isNonnull(); }

  const HeterogeneousProductBase *getBase() const { return impl_.get(); }

  template <typename T>
  const T& get() const {
    if(isNull())
      throw cms::Exception("LogicError") << "HerogeneousProduct is null";

    const auto& ref = *impl_;
    if(typeid(T) != typeid(ref)) {
      throw cms::Exception("LogicError") << "Trying to get HeterogeneousProductImpl " << typeid(T).name() << " but the product contains " << typeid(ref).name();
    }
    return static_cast<const T&>(*impl_);
  }
private:
  std::unique_ptr<HeterogeneousProductBase> impl_;
};

#endif
