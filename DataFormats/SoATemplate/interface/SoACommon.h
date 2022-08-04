#ifndef DataFormats_SoATemplate_interface_SoACommon_h
#define DataFormats_SoATemplate_interface_SoACommon_h

/*
 * Definitions of SoA common parameters for SoA class generators
 */

#include <cstdint>
#include <cassert>
#include <ostream>
#include <tuple>
#include <type_traits>

#include <boost/preprocessor.hpp>

#include "FWCore/Utilities/interface/typedefs.h"

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_DEVICE_ONLY __device__
#define SOA_HOST_DEVICE __host__ __device__
#define SOA_INLINE __forceinline__
#else
#define SOA_HOST_ONLY
#define SOA_DEVICE_ONLY
#define SOA_HOST_DEVICE
#define SOA_INLINE inline __attribute__((always_inline))
#endif

// Exception throwing (or willful crash in kernels)
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define SOA_THROW_OUT_OF_RANGE(A) \
  {                               \
    printf("%s\n", (A));          \
    __trap();                     \
  }
#else
#define SOA_THROW_OUT_OF_RANGE(A) \
  { throw std::out_of_range(A); }
#endif

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1
#define _VALUE_TYPE_EIGEN_COLUMN 2

/* The size type need to be "hardcoded" in the template parameters for classes serialized by ROOT */
#define CMS_SOA_BYTE_SIZE_TYPE std::size_t

namespace cms::soa {

  // size_type for indices. Compatible with ROOT Int_t, but limited to 2G entries
  using size_type = cms_int32_t;
  // byte_size_type for byte counts. Not creating an artificial limit (and not ROOT serialized).
  using byte_size_type = CMS_SOA_BYTE_SIZE_TYPE;

  enum class SoAColumnType {
    scalar = _VALUE_TYPE_SCALAR,
    column = _VALUE_TYPE_COLUMN,
    eigen = _VALUE_TYPE_EIGEN_COLUMN
  };

  namespace RestrictQualify {
    constexpr bool enabled = true;
    constexpr bool disabled = false;
    constexpr bool Default = disabled;
  }  // namespace RestrictQualify

  namespace RangeChecking {
    constexpr bool enabled = true;
    constexpr bool disabled = false;
    constexpr bool Default = disabled;
  }  // namespace RangeChecking

  template <typename T, bool RESTRICT_QUALIFY>
  struct add_restrict {};

  template <typename T>
  struct add_restrict<T, RestrictQualify::enabled> {
    using Value = T;
    using Pointer = T* __restrict__;
    using Reference = T& __restrict__;
    using ConstValue = const T;
    using PointerToConst = const T* __restrict__;
    using ReferenceToConst = const T& __restrict__;
  };

  template <typename T>
  struct add_restrict<T, RestrictQualify::disabled> {
    using Value = T;
    using Pointer = T*;
    using Reference = T&;
    using ConstValue = const T;
    using PointerToConst = const T*;
    using ReferenceToConst = const T&;
  };

  // Forward declarations
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAConstParametersImpl;

  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAParametersImpl;

  // Templated const parameter sets for scalars, columns and Eigen columns
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAConstParametersImpl {
    static constexpr SoAColumnType columnType = COLUMN_TYPE;

    using ValueType = T;
    using ScalarType = T;
    using TupleOrPointerType = const ValueType*;

    // default constructor
    SoAConstParametersImpl() = default;

    // constructor from an address
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(ValueType const* addr) : addr_(addr) {}

    // constructor from a non-const parameter set
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(SoAParametersImpl<columnType, ValueType> const& o)
        : addr_{o.addr_} {}

    static constexpr bool checkAlignment(ValueType* addr, byte_size_type alignment) {
      return reinterpret_cast<intptr_t>(addr) % alignment;
    }

  public:
    // scalar or column
    ValueType const* addr_ = nullptr;
  };

  // Templated const parameter specialisation for Eigen columns
  template <typename T>
  struct SoAConstParametersImpl<SoAColumnType::eigen, T> {
    static constexpr SoAColumnType columnType = SoAColumnType::eigen;

    using ValueType = T;
    using ScalarType = typename T::Scalar;
    using TupleOrPointerType = std::tuple<ScalarType*, byte_size_type>;

    // default constructor
    SoAConstParametersImpl() = default;

    // constructor from individual address and stride
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(ScalarType const* addr, byte_size_type stride)
        : addr_(addr), stride_(stride) {}

    // constructor from address and stride packed in a tuple
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(TupleOrPointerType const& tuple)
        : addr_(std::get<0>(tuple)), stride_(std::get<1>(tuple)) {}

    // constructor from a non-const parameter set
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAConstParametersImpl(SoAParametersImpl<columnType, ValueType> const& o)
        : addr_{o.addr_}, stride_{o.stride_} {}

    static constexpr bool checkAlignment(TupleOrPointerType const& tuple, byte_size_type alignment) {
      const auto& [addr, stride] = tuple;
      return reinterpret_cast<intptr_t>(addr) % alignment;
    }

  public:
    // address and stride
    ScalarType const* addr_ = nullptr;
    byte_size_type stride_ = 0;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAConstParameters_ColumnType {
    template <typename T>
    using DataType = SoAConstParametersImpl<COLUMN_TYPE, T>;
  };

  // Templated parameter sets for scalars, columns and Eigen columns
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAParametersImpl {
    static constexpr SoAColumnType columnType = COLUMN_TYPE;

    using ValueType = T;
    using ScalarType = T;
    using TupleOrPointerType = ValueType*;

    using ConstType = SoAConstParametersImpl<columnType, ValueType>;
    friend ConstType;

    // default constructor
    SoAParametersImpl() = default;

    // constructor from an address
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(ValueType* addr) : addr_(addr) {}

    static constexpr bool checkAlignment(ValueType* addr, byte_size_type alignment) {
      return reinterpret_cast<intptr_t>(addr) % alignment;
    }

  public:
    // scalar or column
    ValueType* addr_ = nullptr;
  };

  // Templated parameter specialisation for Eigen columns
  template <typename T>
  struct SoAParametersImpl<SoAColumnType::eigen, T> {
    static constexpr SoAColumnType columnType = SoAColumnType::eigen;

    using ValueType = T;
    using ScalarType = typename T::Scalar;
    using TupleOrPointerType = std::tuple<ScalarType*, byte_size_type>;

    using ConstType = SoAConstParametersImpl<columnType, ValueType>;
    friend ConstType;

    // default constructor
    SoAParametersImpl() = default;

    // constructor from individual address and stride
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(ScalarType* addr, byte_size_type stride)
        : addr_(addr), stride_(stride) {}

    // constructor from address and stride packed in a tuple
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(TupleOrPointerType const& tuple)
        : addr_(std::get<0>(tuple)), stride_(std::get<1>(tuple)) {}

    static constexpr bool checkAlignment(TupleOrPointerType const& tuple, byte_size_type alignment) {
      const auto& [addr, stride] = tuple;
      return reinterpret_cast<intptr_t>(addr) % alignment;
    }

  public:
    // address and stride
    ScalarType* addr_ = nullptr;
    byte_size_type stride_ = 0;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAParameters_ColumnType {
    template <typename T>
    using DataType = SoAParametersImpl<COLUMN_TYPE, T>;
  };

  // Helper converting a const parameter set to a non-const parameter set, to be used only in the constructor of non-const "element"
  namespace {
    template <typename T>
    constexpr inline std::remove_const_t<T>* non_const_ptr(T* p) {
      return const_cast<std::remove_const_t<T>*>(p);
    }
  }  // namespace

  template <SoAColumnType COLUMN_TYPE, typename T>
  SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl<COLUMN_TYPE, T> const_cast_SoAParametersImpl(
      SoAConstParametersImpl<COLUMN_TYPE, T> const& o) {
    return SoAParametersImpl<COLUMN_TYPE, T>{non_const_ptr(o.addr_)};
  }

  template <typename T>
  SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl<SoAColumnType::eigen, T> const_cast_SoAParametersImpl(
      SoAConstParametersImpl<SoAColumnType::eigen, T> const& o) {
    return SoAParametersImpl<SoAColumnType::eigen, T>{non_const_ptr(o.addr_), o.stride_};
  }

  // Helper template managing the value at index idx within a column.
  // The optional compile time alignment parameter enables informing the
  // compiler of alignment (enforced by caller).
  template <SoAColumnType COLUMN_TYPE,
            typename T,
            byte_size_type ALIGNMENT,
            bool RESTRICT_QUALIFY = RestrictQualify::disabled>
  class SoAValue {
    // Eigen is implemented in a specialization
    static_assert(COLUMN_TYPE != SoAColumnType::eigen);

  public:
    using Restr = add_restrict<T, RESTRICT_QUALIFY>;
    using Val = typename Restr::Value;
    using Ptr = typename Restr::Pointer;
    using Ref = typename Restr::Reference;
    using PtrToConst = typename Restr::PointerToConst;
    using RefToConst = typename Restr::ReferenceToConst;

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, T* col) : idx_(i), col_(col) {}

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, SoAParametersImpl<COLUMN_TYPE, T> params)
        : idx_(i), col_(params.addr_) {}

    SOA_HOST_DEVICE SOA_INLINE Ref operator()() {
      // Ptr type will add the restrict qualifyer if needed
      Ptr col = alignedCol();
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE RefToConst operator()() const {
      // PtrToConst type will add the restrict qualifyer if needed
      PtrToConst col = alignedCol();
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE Ptr operator&() { return &alignedCol()[idx_]; }

    SOA_HOST_DEVICE SOA_INLINE PtrToConst operator&() const { return &alignedCol()[idx_]; }

    /* This was an attempt to implement the syntax
     *
     *     old_value = view.x
     *     view.x = new_value
     *
     * instead of
     *
     *     old_value = view.x()
     *     view.x() = new_value
     *
     *  but it was found to break in some corner cases.
     *  We keep them commented out for the time being.

    SOA_HOST_DEVICE SOA_INLINE operator T&() { return col_[idx_]; }

    template <typename T2>
    SOA_HOST_DEVICE SOA_INLINE Ref operator=(const T2& v) {
      return alignedCol()[idx_] = v;
    }
    */

    using valueType = Val;

    static constexpr auto valueSize = sizeof(T);

  private:
    SOA_HOST_DEVICE SOA_INLINE Ptr alignedCol() const {
      if constexpr (ALIGNMENT) {
        return reinterpret_cast<Ptr>(__builtin_assume_aligned(col_, ALIGNMENT));
      }
      return reinterpret_cast<Ptr>(col_);
    }

    size_type idx_;
    T* col_;
  };

  // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
#ifdef EIGEN_WORLD_VERSION
  // Helper template managing an Eigen-type value at index idx within a column.
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    using Type = C;
    using MapType = Eigen::Map<C, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using CMapType = const Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using Restr = add_restrict<typename C::Scalar, RESTRICT_QUALIFY>;
    using Val = typename Restr::Value;
    using Ptr = typename Restr::Pointer;
    using Ref = typename Restr::Reference;
    using PtrToConst = typename Restr::PointerToConst;
    using RefToConst = typename Restr::ReferenceToConst;

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, typename C::Scalar* col, byte_size_type stride)
        : val_(col + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, SoAParametersImpl<SoAColumnType::eigen, C> params)
        : val_(params.addr_ + i,
               C::RowsAtCompileTime,
               C::ColsAtCompileTime,
               Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          crCol_(params.addr_),
          cVal_(crCol_ + i,
                C::RowsAtCompileTime,
                C::ColsAtCompileTime,
                Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          stride_(params.stride_) {}

    SOA_HOST_DEVICE SOA_INLINE MapType& operator()() { return val_; }

    SOA_HOST_DEVICE SOA_INLINE const CMapType& operator()() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE operator C() { return val_; }

    SOA_HOST_DEVICE SOA_INLINE operator const C() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE C* operator&() { return &val_; }

    SOA_HOST_DEVICE SOA_INLINE const C* operator&() const { return &cVal_; }

    template <class C2>
    SOA_HOST_DEVICE SOA_INLINE MapType& operator=(const C2& v) {
      return val_ = v;
    }

    using ValueType = typename C::Scalar;
    static constexpr auto valueSize = sizeof(C::Scalar);
    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  private:
    MapType val_;
    const Ptr crCol_;
    CMapType cVal_;
    byte_size_type stride_;
  };
#else
  // Raise a compile-time error
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // Helper template managing a const value at index idx within a column.
  template <SoAColumnType COLUMN_TYPE,
            typename T,
            byte_size_type ALIGNMENT,
            bool RESTRICT_QUALIFY = RestrictQualify::disabled>
  class SoAConstValue {
    // Eigen is implemented in a specialization
    static_assert(COLUMN_TYPE != SoAColumnType::eigen);

  public:
    using Restr = add_restrict<T, RESTRICT_QUALIFY>;
    using Val = typename Restr::Value;
    using Ptr = typename Restr::Pointer;
    using Ref = typename Restr::Reference;
    using PtrToConst = typename Restr::PointerToConst;
    using RefToConst = typename Restr::ReferenceToConst;
    using Params = SoAParametersImpl<COLUMN_TYPE, T>;
    using ConstParams = SoAConstParametersImpl<COLUMN_TYPE, T>;

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, const T* col) : idx_(i), col_(col) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAParametersImpl<COLUMN_TYPE, T> params)
        : idx_(i), col_(params.addr_) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAConstParametersImpl<COLUMN_TYPE, T> params)
        : idx_(i), col_(params.addr_) {}

    SOA_HOST_DEVICE SOA_INLINE RefToConst operator()() const {
      // Ptr type will add the restrict qualifyer if needed
      PtrToConst col = alignedCol();
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE const T* operator&() const { return &alignedCol()[idx_]; }

    /* This was an attempt to implement the syntax
     *
     *     old_value = view.x
     *
     * instead of
     *
     *     old_value = view.x()
     *
     *  but it was found to break in some corner cases.
     *  We keep them commented out for the time being.

    SOA_HOST_DEVICE SOA_INLINE operator T&() { return col_[idx_]; }
    */

    using valueType = T;
    static constexpr auto valueSize = sizeof(T);

  private:
    SOA_HOST_DEVICE SOA_INLINE PtrToConst alignedCol() const {
      if constexpr (ALIGNMENT) {
        return reinterpret_cast<PtrToConst>(__builtin_assume_aligned(col_, ALIGNMENT));
      }
      return reinterpret_cast<PtrToConst>(col_);
    }

    size_type idx_;
    const T* col_;
  };

  // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
#ifdef EIGEN_WORLD_VERSION
  // Helper template managing a const Eigen-type value at index idx within a column.
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    using Type = C;
    using CMapType = Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>>;
    using RefToConst = const CMapType&;
    using ConstParams = SoAConstParametersImpl<SoAColumnType::eigen, C>;

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, typename C::Scalar* col, byte_size_type stride)
        : crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAConstParametersImpl<SoAColumnType::eigen, C> params)
        : crCol_(params.addr_),
          cVal_(crCol_ + i,
                C::RowsAtCompileTime,
                C::ColsAtCompileTime,
                Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          stride_(params.stride_) {}

    SOA_HOST_DEVICE SOA_INLINE const CMapType& operator()() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE operator const C() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE const C* operator&() const { return &cVal_; }

    using ValueType = typename C::Scalar;
    static constexpr auto valueSize = sizeof(C::Scalar);

    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  private:
    const typename C::Scalar* __restrict__ crCol_;
    CMapType cVal_;
    byte_size_type stride_;
  };
#else
  // Raise a compile-time error
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // Helper template to avoid commas inside macros
#ifdef EIGEN_WORLD_VERSION
  template <class C>
  struct EigenConstMapMaker {
    using Type = Eigen::Map<const C, Eigen::AlignmentType::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;

    class DataHolder {
    public:
      DataHolder(const typename C::Scalar* data) : data_(data) {}

      EigenConstMapMaker::Type withStride(byte_size_type stride) {
        return EigenConstMapMaker::Type(
            data_, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride));
      }

    private:
      const typename C::Scalar* const data_;
    };

    static DataHolder withData(const typename C::Scalar* data) { return DataHolder(data); }
  };
#else
  template <class C>
  struct EigenConstMapMaker {
    // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // Helper function to compute aligned size
  constexpr inline byte_size_type alignSize(byte_size_type size, byte_size_type alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
  }

}  // namespace cms::soa

#define SOA_SCALAR(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME)
#define SOA_COLUMN(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME)
#define SOA_EIGEN_COLUMN(TYPE, NAME) (_VALUE_TYPE_EIGEN_COLUMN, TYPE, NAME)

/* Iterate on the macro MACRO and return the result as a comma separated list */
#define _ITERATE_ON_ALL_COMMA(MACRO, DATA, ...) \
  BOOST_PP_TUPLE_ENUM(BOOST_PP_SEQ_TO_TUPLE(_ITERATE_ON_ALL(MACRO, DATA, __VA_ARGS__)))

/* Iterate MACRO on all elements */
#define _ITERATE_ON_ALL(MACRO, DATA, ...) BOOST_PP_SEQ_FOR_EACH(MACRO, DATA, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

/* Switch on macros depending on scalar / column type */
#define _SWITCH_ON_TYPE(VALUE_TYPE, IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN) \
  BOOST_PP_IF(                                                             \
      BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_SCALAR),                      \
      IF_SCALAR,                                                           \
      BOOST_PP_IF(                                                         \
          BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_COLUMN),                  \
          IF_COLUMN,                                                       \
          BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_EIGEN_COLUMN), IF_EIGEN_COLUMN, BOOST_PP_EMPTY())))

namespace cms::soa {

  /* Column accessors: templates implementing the global accesors (soa::x() and soa::x(index) */
  enum class SoAAccessType : bool { mutableAccess, constAccess };

  template <typename, SoAColumnType, SoAAccessType>
  struct SoAColumnAccessorsImpl {};

  // TODO from Eric Cano:
  //   - add alignment support
  //   - SFINAE-based const/non const variants

  // Column
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::mutableAccess> {
    //SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(T* baseAddress) : baseAddress_(baseAddress) {}
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::column, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T* operator()() { return params_.addr_; }
    using NoParamReturnType = T*;
    SOA_HOST_DEVICE SOA_INLINE T& operator()(size_type index) { return params_.addr_[index]; }

  private:
    SoAParametersImpl<SoAColumnType::column, T> params_;
  };

  // Const column
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::constAccess> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::column, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE const T* operator()() const { return params_.addr_; }
    using NoParamReturnType = const T*;
    SOA_HOST_DEVICE SOA_INLINE T operator()(size_type index) const { return params_.addr_[index]; }

  private:
    SoAConstParametersImpl<SoAColumnType::column, T> params_;
  };

  // Scalar
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::mutableAccess> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T& operator()() { return *params_.addr_; }
    using NoParamReturnType = T&;
    SOA_HOST_DEVICE SOA_INLINE void operator()(size_type index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Const scalar
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::constAccess> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T operator()() const { return *params_.addr_; }
    using NoParamReturnType = T;
    SOA_HOST_DEVICE SOA_INLINE void operator()(size_type index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAConstParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Eigen-type
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::mutableAccess> {
    //SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(T* baseAddress) : baseAddress_(baseAddress) {}
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::eigen, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE typename T::Scalar* operator()() { return params_.addr_; }
    using NoParamReturnType = typename T::Scalar*;
    //SOA_HOST_DEVICE SOA_INLINE T& operator()(size_type index) { return params_.addr_[index]; }

  private:
    SoAParametersImpl<SoAColumnType::eigen, T> params_;
  };

  // Const Eigen-type
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::constAccess> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::eigen, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE const typename T::Scalar* operator()() const { return params_.addr_; }
    using NoParamReturnType = typename T::Scalar*;
    //SOA_HOST_DEVICE SOA_INLINE T operator()(size_type index) const { return params_.addr_[index]; }

  private:
    SoAConstParametersImpl<SoAColumnType::eigen, T> params_;
  };

  /* A helper template stager to avoid commas inside macros */
  template <typename T>
  struct SoAAccessors {
    template <auto columnType>
    struct ColumnType {
      template <auto accessType>
      struct AccessType : public SoAColumnAccessorsImpl<T, columnType, accessType> {
        using SoAColumnAccessorsImpl<T, columnType, accessType>::SoAColumnAccessorsImpl;
      };
    };
  };

  /* Enum parameters allowing templated control of layout/view behaviors */
  /* Alignment enforcement verifies every column is aligned, and
   * hints the compiler that it can expect column pointers to be aligned */
  struct AlignmentEnforcement {
    static constexpr bool relaxed = false;
    static constexpr bool enforced = true;
  };

  struct CacheLineSize {
    static constexpr byte_size_type NvidiaGPU = 128;
    static constexpr byte_size_type IntelCPU = 64;
    static constexpr byte_size_type AMDCPU = 64;
    static constexpr byte_size_type ARMCPU = 64;
    static constexpr byte_size_type defaultSize = NvidiaGPU;
  };

}  // namespace cms::soa

// Small wrapper for stream insertion of SoA printing
template <typename SOA,
          typename SFINAE =
              typename std::enable_if_t<std::is_invocable_v<decltype(&SOA::soaToStreamInternal), SOA&, std::ostream&>>>
SOA_HOST_ONLY std::ostream& operator<<(std::ostream& os, const SOA& soa) {
  soa.soaToStreamInternal(os);
  return os;
}

#endif  // DataFormats_SoATemplate_interface_SoACommon_h
