#ifndef DataFormats_SoATemplate_interface_SoACommon_h
#define DataFormats_SoATemplate_interface_SoACommon_h

/*
 * Definitions of SoA common parameters for SoA class generators
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <memory>
#include <ostream>
#include <sstream>
#include <span>
#include <source_location>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/preprocessor.hpp>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_runtime_api.h>
#endif

#include "FWCore/Utilities/interface/typedefs.h"

// CUDA/ROCm attributes
#if defined(__CUDACC__) || defined(__HIPCC__)
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

#define SOA_THROW_OUT_OF_RANGE(A, I, R)                                     \
  {                                                                         \
    if constexpr (decltype(I)::mode == cms::soa::RangeChecking::extended) { \
      printf("%s: index %d out of range %d at file %s line %d\n",           \
             (A),                                                           \
             (I.value_),                                                    \
             (R),                                                           \
             (I.location_.file_name()),                                     \
             (I.location_.line()));                                         \
    } else {                                                                \
      printf("%s: index %d out of range %d\n", (A), (I.value_), (R));       \
    }                                                                       \
    __trap();                                                               \
  }

#elif defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__)

#define SOA_THROW_OUT_OF_RANGE(A, I, R)                                     \
  {                                                                         \
    if constexpr (decltype(I)::mode == cms::soa::RangeChecking::extended) { \
      printf("%s: index %d out of range %d at file %s line %d\n",           \
             (A),                                                           \
             (I.value_),                                                    \
             (R),                                                           \
             (I.location_.file_name()),                                     \
             (I.location_.line()));                                         \
    } else {                                                                \
      printf("%s: index %d out of range %d\n", (A), (I.value_), (R));       \
    }                                                                       \
    abort();                                                                \
  }

#else
#define SOA_THROW_OUT_OF_RANGE(A, I, R)                                       \
  {                                                                           \
    cms::soa::detail::throwOutOfRangeError<decltype(I)::mode>((A), (I), (R)); \
  }
#endif

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1
#define _VALUE_TYPE_EIGEN_COLUMN 2
#define _VALUE_TYPE_METHOD 3
#define _VALUE_TYPE_CONST_METHOD 4
#define _VALUE_TYPE_BLOCK 5
#define _VALUE_TYPE_VIEW_METHOD 6
#define _VALUE_TYPE_CONST_VIEW_METHOD 7

/* declare the value of last valid column */
#define _VALUE_LAST_COLUMN_TYPE _VALUE_TYPE_EIGEN_COLUMN

/* declare a macro useful for passing a valid but not used value*/
#define _VALUE_TYPE_UNUSED BOOST_PP_LIMIT_MAG

/* Properties of the value types as lookup tables, used instead of BOOST_PP_GREATER / LESS_EQUAL / EQUAL: Boost.PP
   compares numbers by looped subtraction, hundreds of macro expansions for each element of each generated method.
   Each table has one entry per value type: a new type without entries fails to compile (undeclared _..._<type>). */
#if _VALUE_TYPE_SCALAR != 0 || _VALUE_TYPE_COLUMN != 1 || _VALUE_TYPE_EIGEN_COLUMN != 2 || _VALUE_TYPE_METHOD != 3 || \
    _VALUE_TYPE_CONST_METHOD != 4 || _VALUE_TYPE_BLOCK != 5 || _VALUE_TYPE_VIEW_METHOD != 6 ||                        \
    _VALUE_TYPE_CONST_VIEW_METHOD != 7 || _VALUE_LAST_COLUMN_TYPE != 2 || _VALUE_TYPE_UNUSED != 256
#error "The SoA value types changed: update the lookup tables _IS_NOT_COLUMN_*, _IS_SCALAR_*, _IS_BLOCK_TYPE_*"
#endif

/* 1 if the value type is not a data column (scalar, column or Eigen column): VALUE_TYPE > _VALUE_LAST_COLUMN_TYPE */
#define _IS_NOT_COLUMN(VALUE_TYPE) BOOST_PP_CAT(_IS_NOT_COLUMN_, VALUE_TYPE)
#define _IS_NOT_COLUMN_0 0
#define _IS_NOT_COLUMN_1 0
#define _IS_NOT_COLUMN_2 0
#define _IS_NOT_COLUMN_3 1
#define _IS_NOT_COLUMN_4 1
#define _IS_NOT_COLUMN_5 1
#define _IS_NOT_COLUMN_6 1
#define _IS_NOT_COLUMN_7 1
#define _IS_NOT_COLUMN_256 1

/* 1 if the value type is a scalar: VALUE_TYPE == _VALUE_TYPE_SCALAR */
#define _IS_SCALAR(VALUE_TYPE) BOOST_PP_CAT(_IS_SCALAR_, VALUE_TYPE)
#define _IS_SCALAR_0 1
#define _IS_SCALAR_1 0
#define _IS_SCALAR_2 0
#define _IS_SCALAR_3 0
#define _IS_SCALAR_4 0
#define _IS_SCALAR_5 0
#define _IS_SCALAR_6 0
#define _IS_SCALAR_7 0
#define _IS_SCALAR_256 0

/* 1 if the value type is a block or lower (see _IS_BLOCK): VALUE_TYPE <= _VALUE_TYPE_BLOCK */
#define _IS_BLOCK_TYPE(VALUE_TYPE) BOOST_PP_CAT(_IS_BLOCK_TYPE_, VALUE_TYPE)
#define _IS_BLOCK_TYPE_0 1
#define _IS_BLOCK_TYPE_1 1
#define _IS_BLOCK_TYPE_2 1
#define _IS_BLOCK_TYPE_3 1
#define _IS_BLOCK_TYPE_4 1
#define _IS_BLOCK_TYPE_5 1
#define _IS_BLOCK_TYPE_6 0
#define _IS_BLOCK_TYPE_7 0
#define _IS_BLOCK_TYPE_256 0

/* The size type need to be "hardcoded" in the template parameters for classes serialized by ROOT */
/* In practice, using a typedef as a template parameter to the Layout or its ViewTemplateFreeParams member
 * declaration fails ROOT dictionary generation.  */
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
    constexpr bool Default = enabled;
  }  // namespace RestrictQualify

  namespace RangeChecking {

    enum class Mode { Disabled, Enabled, Extended };

    constexpr Mode disabled = Mode::Disabled;
    constexpr Mode enabled = Mode::Enabled;
    constexpr Mode extended = Mode::Extended;

    constexpr Mode Default = enabled;

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

  // Matryoshka template to avoid commas inside macros
  template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT>
  struct LayoutParameters {
    template <bool ALIGNMENT_ENFORCEMENT>
    struct AlignmentEnforcement {
      template <template <CMS_SOA_BYTE_SIZE_TYPE, bool> typename T>
      using Layout = T<ALIGNMENT, ALIGNMENT_ENFORCEMENT>;
    };
  };

  // Stride type for the Eigen::Map objects that give access to the elements of an Eigen column.
  //
  // The elements of a matrix in an Eigen column are stored "stride" positions apart, with the stride known only at
  // runtime. Eigen describes this layout with Eigen::InnerStride<Eigen::Dynamic>.
  // See https://github.com/PX4/eigen/blob/master/Eigen/src/Core/Stride.h for more information
  // This class replaces it so that the SoA headers do not depend on Eigen;
  // the Map type itself comes from C::StridedMapType<DynamicInnerStride>.
  // The Eigen stride class uses std::ptrdiff_t for indexing by default, hence used here to match what eigen expects
  // See https://github.com/PX4/eigen/blob/master/Eigen/src/Core/util/Macros.h#L336
  class DynamicInnerStride {
  public:
    enum {
      InnerStrideAtCompileTime = -1,  // -1 means that the inner stride can be set dynamically at runtime
      OuterStrideAtCompileTime = 0    // 0 means that the outer stride is hard fixed to 0 and can not be changed
    };

    SOA_HOST_DEVICE SOA_INLINE constexpr explicit DynamicInnerStride(byte_size_type inner) noexcept
        : inner_(static_cast<std::ptrdiff_t>(inner)) {}

    SOA_HOST_DEVICE SOA_INLINE constexpr std::ptrdiff_t outer() const noexcept { return OuterStrideAtCompileTime; }
    SOA_HOST_DEVICE SOA_INLINE constexpr std::ptrdiff_t inner() const noexcept { return inner_; }

  private:
    const std::ptrdiff_t inner_;
  };

  // Templated parameter sets for scalars, columns and Eigen columns
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAParametersImpl {
    static constexpr SoAColumnType columnType = COLUMN_TYPE;

    using ValueType = std::remove_cvref_t<T>;
    using ScalarType = T;

    using ConstType = SoAParametersImpl<columnType, const ValueType>;
    friend ConstType;

    // default constructor
    SoAParametersImpl() = default;

    // constructor from address and size
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(ScalarType* addr) : addr_(addr) {}

    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(SoAParametersImpl<COLUMN_TYPE, ValueType> const& params)
        : addr_(const_cast<ScalarType*>(params.addr_)) {}

    SOA_HOST_DEVICE SOA_INLINE ScalarType* data() const { return addr_; }

  public:
    // scalar or column
    ScalarType* addr_ = nullptr;
  };

  // Templated parameter specialisation for Eigen columns
  template <typename T>
  struct SoAParametersImpl<SoAColumnType::eigen, T> {
    static constexpr SoAColumnType columnType = SoAColumnType::eigen;

    using ValueType = std::remove_cvref_t<T>;
    using ScalarType = typename T::Scalar;

    using ConstType = SoAParametersImpl<columnType, const ValueType>;
    friend ConstType;

    // default constructor
    SoAParametersImpl() = default;

    // constructor from individual address, stride and size
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(ScalarType* addr, byte_size_type stride)
        : addr_(addr), stride_(stride) {}
    SOA_HOST_DEVICE SOA_INLINE constexpr SoAParametersImpl(
        SoAParametersImpl<SoAColumnType::eigen, ValueType> const& params)
        : addr_(const_cast<ScalarType*>(params.addr_)), stride_(params.stride_) {}

    SOA_HOST_DEVICE SOA_INLINE ScalarType* data() const { return addr_; }
    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  public:
    // address, stride and size
    ScalarType* addr_ = nullptr;
    byte_size_type stride_ = 0;
  };

  template <SoAColumnType COLUMN_TYPE, typename T>
    requires std::same_as<std::remove_const_t<T>, T>
  using SoAConstParametersImpl = SoAParametersImpl<COLUMN_TYPE, const T>;

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAConstParameters_ColumnType {
    template <typename T>
    using DataType = SoAConstParametersImpl<COLUMN_TYPE, T>;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAParameters_ColumnType {
    template <typename T>
    using DataType = SoAParametersImpl<COLUMN_TYPE, T>;
  };

  template <typename COLUMN>
  struct Tuple {
    using Type = std::tuple<COLUMN, size_type>;
  };

  template <typename COLUMN>
  struct ConstTuple {
    using Type = std::tuple<typename COLUMN::ConstType, size_type>;
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
      Ptr col = col_;
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE RefToConst operator()() const {
      // PtrToConst type will add the restrict qualifyer if needed
      PtrToConst col = col_;
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE Ptr operator&() { return &col_[idx_]; }

    SOA_HOST_DEVICE SOA_INLINE PtrToConst operator&() const { return &col_[idx_]; }

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
      return col_[idx_] = v;
    }
    */

    using valueType = Val;

    static constexpr auto valueSize = sizeof(T);

  private:
    size_type idx_;
    T* col_;
  };

  // Helper template managing an Eigen-type value at index idx within a column.
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    using Type = C;
    using StrideT = DynamicInnerStride;
    using MapType = typename C::template StridedMapType<StrideT>::type;
    using CMapType = const typename C::template StridedConstMapType<StrideT>::type;
    using Restr = add_restrict<typename C::Scalar, RESTRICT_QUALIFY>;
    using Val = typename Restr::Value;
    using Ptr = typename Restr::Pointer;
    using Ref = typename Restr::Reference;
    using PtrToConst = typename Restr::PointerToConst;
    using RefToConst = typename Restr::ReferenceToConst;

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, typename C::Scalar* col, byte_size_type stride)
        : val_(col + i, C::RowsAtCompileTime, C::ColsAtCompileTime, StrideT(stride)),
          crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, StrideT(stride)),
          stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE SoAValue(size_type i, SoAParametersImpl<SoAColumnType::eigen, C> params)
        : val_(params.addr_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, StrideT(params.stride_)),
          crCol_(params.addr_),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, StrideT(params.stride_)),
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
    static constexpr auto valueSize = sizeof(typename C::Scalar);
    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  private:
    MapType val_;
    const Ptr crCol_;
    CMapType cVal_;
    byte_size_type stride_;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAValue_ColumnType {
    template <typename T>
    struct DataType {
      template <byte_size_type ALIGNMENT>
      struct Alignment {
        template <bool RESTRICT_QUALIFY>
        using Value = SoAValue<COLUMN_TYPE, T, ALIGNMENT, RESTRICT_QUALIFY>;
      };
    };
  };

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
      PtrToConst col = col_;
      return col[idx_];
    }

    SOA_HOST_DEVICE SOA_INLINE const T* operator&() const { return &col_[idx_]; }

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
    size_type idx_;
    const T* col_;
  };

  // Helper template managing a const Eigen-type value at index idx within a column.
  template <class C, byte_size_type ALIGNMENT, bool RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    using Type = C;
    using StrideT = DynamicInnerStride;
    using CMapType = const typename C::template StridedConstMapType<StrideT>::type;
    using RefToConst = const CMapType&;
    using ConstParams = SoAConstParametersImpl<SoAColumnType::eigen, C>;

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, typename C::Scalar* col, byte_size_type stride)
        : crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, StrideT(stride)),
          stride_(stride) {}

    SOA_HOST_DEVICE SOA_INLINE SoAConstValue(size_type i, SoAConstParametersImpl<SoAColumnType::eigen, C> params)
        : crCol_(params.addr_),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, StrideT(params.stride_)),
          stride_(params.stride_) {}

    SOA_HOST_DEVICE SOA_INLINE const CMapType& operator()() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE operator const C() const { return cVal_; }

    SOA_HOST_DEVICE SOA_INLINE const C* operator&() const { return &cVal_; }

    using ValueType = typename C::Scalar;
    static constexpr auto valueSize = sizeof(typename C::Scalar);

    SOA_HOST_DEVICE SOA_INLINE byte_size_type stride() const { return stride_; }

  private:
    const typename C::Scalar* __restrict__ crCol_;
    CMapType cVal_;
    byte_size_type stride_;
  };

  // Matryoshka template to avoid commas inside macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAConstValue_ColumnType {
    template <typename T>
    struct DataType {
      template <byte_size_type ALIGNMENT>
      struct Alignment {
        template <bool RESTRICT_QUALIFY>
        using ConstValue = SoAConstValue<COLUMN_TYPE, T, ALIGNMENT, RESTRICT_QUALIFY>;
      };
    };
  };

  // Helper function to compute aligned size
  // this is an integer division -> it rounds size to the next multiple of alignment
  constexpr inline byte_size_type alignSize(byte_size_type size, byte_size_type alignment) {
    assert(alignment > 0 && "Alignment for SoA must be > 0");
    return ((size + alignment - 1) / alignment) * alignment;
  }

}  // namespace cms::soa

#define SOA_SCALAR(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME, ~)
#define SOA_COLUMN(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME, ~)
#define SOA_EIGEN_COLUMN(TYPE, NAME) (_VALUE_TYPE_EIGEN_COLUMN, TYPE, NAME, ~)
#define SOA_ELEMENT_METHODS(...) (_VALUE_TYPE_METHOD, _, _, (__VA_ARGS__))
#define SOA_CONST_ELEMENT_METHODS(...) (_VALUE_TYPE_CONST_METHOD, _, _, (__VA_ARGS__))
#define SOA_BLOCK(NAME, LAYOUT_NAME) (_VALUE_TYPE_BLOCK, NAME, LAYOUT_NAME)
#define SOA_VIEW_METHODS(...) (_VALUE_TYPE_VIEW_METHOD, _, _, (__VA_ARGS__))
#define SOA_CONST_VIEW_METHODS(...) (_VALUE_TYPE_CONST_VIEW_METHOD, _, _, (__VA_ARGS__))

/* Macro generating customized methods for the element */
#define GENERATE_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_METHOD), \
              BOOST_PP_TUPLE_ELEM(3, FIELD),                                     \
              BOOST_PP_EMPTY())

/* Macro generating customized methods for the const element*/
#define GENERATE_CONST_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_CONST_METHOD), \
              BOOST_PP_TUPLE_ELEM(3, FIELD),                                           \
              BOOST_PP_EMPTY())

/* Macro generating customized methods for the View of an SoA */
#define GENERATE_VIEW_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_VIEW_METHOD), \
              BOOST_PP_TUPLE_ELEM(3, FIELD),                                          \
              BOOST_PP_EMPTY())

/* Macro generating customized methods for the ConstView of an SoA */
#define GENERATE_CONST_VIEW_METHODS(R, DATA, FIELD)                                         \
  BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, FIELD), _VALUE_TYPE_CONST_VIEW_METHOD), \
              BOOST_PP_TUPLE_ELEM(3, FIELD),                                                \
              BOOST_PP_EMPTY())

/* Preprocessing loop for managing functions generation: only macros containing valid content are expanded */
#define ENUM_FOR_PRED(R, STATE) BOOST_PP_LESS(BOOST_PP_TUPLE_ELEM(0, STATE), BOOST_PP_TUPLE_ELEM(1, STATE))

#define ENUM_FOR_OP(R, STATE) \
  (BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(0, STATE)), BOOST_PP_TUPLE_ELEM(1, STATE), BOOST_PP_TUPLE_ELEM(2, STATE))

#define ENUM_FOR_MACRO(R, STATE) \
  BOOST_PP_TUPLE_ENUM(BOOST_PP_SEQ_ELEM(BOOST_PP_TUPLE_ELEM(0, STATE), BOOST_PP_TUPLE_ELEM(2, STATE)))

#define ENUM_IF_VALID(...)                                                                      \
  BOOST_PP_FOR((0, BOOST_PP_VARIADIC_SIZE(__VA_ARGS__), BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)), \
               ENUM_FOR_PRED,                                                                   \
               ENUM_FOR_OP,                                                                     \
               ENUM_FOR_MACRO)

/* Iterate on the macro MACRO and return the result as a comma separated list: the sequence (a)(b)(c) becomes a, b, c
   with one macro expansion per element (as BOOST_PP_TUPLE_ENUM(BOOST_PP_SEQ_TO_TUPLE(...)), at a lower cost) */
#define _ITERATE_ON_ALL_COMMA(MACRO, DATA, ...) _SEQ_TO_LIST(_ITERATE_ON_ALL(MACRO, DATA, __VA_ARGS__))

/* (a)(b)(c) -> a _SEQ_TO_LIST_A(b) ... -> a, b, c _SEQ_TO_LIST_A; the trailing macro name is pasted with _END */
#define _SEQ_TO_LIST(SEQ) _SEQ_TO_LIST_I(_SEQ_TO_LIST_FIRST SEQ)
#define _SEQ_TO_LIST_I(...) _SEQ_TO_LIST_II(__VA_ARGS__)
#define _SEQ_TO_LIST_II(...) __VA_ARGS__##_END
#define _SEQ_TO_LIST_FIRST(...) __VA_ARGS__ _SEQ_TO_LIST_A
#define _SEQ_TO_LIST_A(...) , __VA_ARGS__ _SEQ_TO_LIST_B
#define _SEQ_TO_LIST_B(...) , __VA_ARGS__ _SEQ_TO_LIST_A
#define _SEQ_TO_LIST_FIRST_END
#define _SEQ_TO_LIST_A_END
#define _SEQ_TO_LIST_B_END

/* Iterate MACRO on all elements: MACRO(R, DATA, ELEMENT) for each element, as BOOST_PP_SEQ_FOR_EACH did on
   BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__), but with one macro expansion per element: BOOST_PP_SEQ_FOR_EACH carries the
   remaining sequence through several macro layers per step, a quadratic cost in the macro locations tracked by the
   compiler. The iterated macros do not use R (it is passed as ~). Up to 128 elements (Boost.PP before: 64).  */
// clang-format off
#define _ITERATE_ON_ALL(MACRO, DATA, ...) \
  BOOST_PP_CAT(_ITERATE_ON_ALL_, _ITERATE_ON_ALL_SIZE(__VA_ARGS__))(MACRO, DATA, __VA_ARGS__)

/* number of elements, or _TOO_MANY above the capacity (up to 144 elements, then a token-pasting error) */
#define _ITERATE_ON_ALL_SIZE(...) _ITERATE_ON_ALL_SIZE_I(__VA_ARGS__, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, \
    _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, _TOO_MANY, \
    _TOO_MANY, _TOO_MANY, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, \
    110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, \
    84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, \
    56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, \
    28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _ITERATE_ON_ALL_SIZE_I(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, \
    _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, \
    _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, \
    _89, _90, _91, _92, _93, _94, _95, _96, _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107, _108, _109, \
    _110, _111, _112, _113, _114, _115, _116, _117, _118, _119, _120, _121, _122, _123, _124, _125, _126, _127, _128, \
    _129, _130, _131, _132, _133, _134, _135, _136, _137, _138, _139, _140, _141, _142, _143, _144, N, ...) N
#define _ITERATE_ON_ALL__TOO_MANY(MACRO, DATA, ...) \
  _Pragma("GCC error \"GENERATE_SOA_LAYOUT / GENERATE_SOA_BLOCKS: more than 128 elements, see _ITERATE_ON_ALL\"")
#define _ITERATE_ON_ALL_1(M, D, X) M(~, D, X)
#define _ITERATE_ON_ALL_2(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_1(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_3(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_2(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_4(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_3(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_5(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_4(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_6(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_5(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_7(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_6(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_8(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_7(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_9(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_8(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_10(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_9(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_11(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_10(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_12(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_11(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_13(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_12(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_14(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_13(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_15(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_14(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_16(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_15(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_17(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_16(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_18(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_17(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_19(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_18(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_20(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_19(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_21(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_20(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_22(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_21(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_23(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_22(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_24(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_23(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_25(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_24(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_26(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_25(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_27(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_26(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_28(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_27(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_29(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_28(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_30(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_29(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_31(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_30(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_32(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_31(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_33(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_32(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_34(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_33(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_35(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_34(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_36(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_35(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_37(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_36(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_38(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_37(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_39(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_38(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_40(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_39(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_41(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_40(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_42(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_41(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_43(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_42(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_44(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_43(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_45(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_44(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_46(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_45(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_47(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_46(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_48(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_47(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_49(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_48(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_50(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_49(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_51(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_50(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_52(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_51(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_53(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_52(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_54(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_53(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_55(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_54(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_56(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_55(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_57(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_56(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_58(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_57(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_59(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_58(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_60(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_59(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_61(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_60(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_62(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_61(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_63(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_62(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_64(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_63(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_65(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_64(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_66(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_65(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_67(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_66(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_68(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_67(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_69(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_68(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_70(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_69(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_71(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_70(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_72(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_71(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_73(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_72(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_74(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_73(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_75(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_74(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_76(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_75(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_77(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_76(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_78(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_77(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_79(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_78(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_80(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_79(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_81(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_80(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_82(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_81(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_83(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_82(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_84(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_83(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_85(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_84(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_86(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_85(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_87(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_86(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_88(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_87(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_89(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_88(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_90(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_89(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_91(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_90(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_92(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_91(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_93(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_92(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_94(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_93(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_95(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_94(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_96(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_95(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_97(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_96(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_98(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_97(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_99(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_98(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_100(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_99(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_101(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_100(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_102(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_101(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_103(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_102(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_104(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_103(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_105(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_104(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_106(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_105(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_107(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_106(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_108(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_107(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_109(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_108(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_110(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_109(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_111(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_110(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_112(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_111(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_113(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_112(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_114(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_113(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_115(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_114(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_116(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_115(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_117(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_116(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_118(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_117(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_119(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_118(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_120(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_119(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_121(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_120(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_122(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_121(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_123(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_122(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_124(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_123(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_125(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_124(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_126(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_125(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_127(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_126(M, D, __VA_ARGS__)
#define _ITERATE_ON_ALL_128(M, D, X, ...) M(~, D, X) _ITERATE_ON_ALL_127(M, D, __VA_ARGS__)
// clang-format on

/* Switch on macros depending on scalar / column type (nothing for the other value types) */
#define _SWITCH_ON_TYPE(VALUE_TYPE, IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN) \
  BOOST_PP_CAT(_SWITCH_ON_TYPE_, VALUE_TYPE)(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)
#define _SWITCH_ON_TYPE_0(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN) IF_SCALAR
#define _SWITCH_ON_TYPE_1(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN) IF_COLUMN
#define _SWITCH_ON_TYPE_2(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN) IF_EIGEN_COLUMN
#define _SWITCH_ON_TYPE_3(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)
#define _SWITCH_ON_TYPE_4(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)
#define _SWITCH_ON_TYPE_5(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)
#define _SWITCH_ON_TYPE_6(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)
#define _SWITCH_ON_TYPE_7(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)
#define _SWITCH_ON_TYPE_256(IF_SCALAR, IF_COLUMN, IF_EIGEN_COLUMN)

// Extract the type, name and layout from a block specification
#define _BLOCK_GET_TYPE(SPEC) BOOST_PP_TUPLE_ELEM(0, SPEC)
#define _BLOCK_GET_NAME(SPEC) BOOST_PP_TUPLE_ELEM(1, SPEC)
#define _BLOCK_GET_LAYOUT(SPEC) BOOST_PP_TUPLE_ELEM(2, SPEC)

// Check if argument is a block specification
#define _IS_BLOCK(SPEC) _IS_BLOCK_TYPE(_BLOCK_GET_TYPE(SPEC))

// Execute MACRO if specification is a block specification, otherwise do nothing
#define _EXEC_IF_BLOCK(SPEC, MACRO, ARGS) BOOST_PP_IF(_IS_BLOCK(SPEC), MACRO ARGS, BOOST_PP_EMPTY())

#define _APPLY_ONLY_FOR_SCALAR(VALUE_TYPE, CODE) BOOST_PP_IF(_IS_SCALAR(VALUE_TYPE), CODE, )

#define _APPLY_FOR_NON_SCALAR(VALUE_TYPE, CODE) BOOST_PP_IF(_IS_SCALAR(VALUE_TYPE), , CODE)

/* Produces text input token if input sequence is not empty */
#define _APPEND_TOKEN_1(PARAM_NAME)
#define _APPEND_TOKEN_0(PARAM_NAME) PARAM_NAME
#define _APPEND_TOKEN(SEQ, PARAM_NAME) BOOST_PP_CAT(_APPEND_TOKEN_, BOOST_PP_IS_EMPTY(SEQ))(PARAM_NAME)

/* Appends comma if input sequence is not empty */
#define _APPEND_COMMA_1(SEQ)
#define _APPEND_COMMA_0(SEQ) , BOOST_PP_SEQ_ENUM(SEQ)
#define _APPEND_COMMA(SEQ) BOOST_PP_CAT(_APPEND_COMMA_, BOOST_PP_IS_EMPTY(SEQ))(SEQ)

/* Appends list initializer token ":" if input sequence is not empty */
#define _APPEND_LIST_INIT_1(SEQ)
#define _APPEND_LIST_INIT_0(SEQ) : BOOST_PP_SEQ_ENUM(SEQ)
#define _APPEND_LIST_INIT(SEQ) BOOST_PP_CAT(_APPEND_LIST_INIT_, BOOST_PP_IS_EMPTY(SEQ))(SEQ)

namespace cms::soa {

  /* Column accessors: templates implementing the global accesors (soa::x() and soa::x(index) */
  enum class SoAAccessType : bool { mutableAccess, constAccess };

  template <typename, SoAColumnType, SoAAccessType, byte_size_type, bool>
  struct SoAColumnAccessorsImpl {};

  // TODO from Eric Cano:
  //   - add alignment support
  //   - SFINAE-based const/non const variants

  // Column
  template <typename T, byte_size_type alignment, bool restrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::mutableAccess, alignment, restrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::column, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<T> operator()() { return std::span<T>(params_.addr_, size_); }

    using NoParamReturnType = std::span<T>;
    using ParamReturnType = T&;
    SOA_HOST_DEVICE SOA_INLINE T& operator()(size_type index) { return params_.addr_[index]; }

  private:
    SoAParametersImpl<SoAColumnType::column, T> params_;
    size_type size_ = 0;
  };

  // Const column
  template <typename T, byte_size_type alignment, bool restrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::constAccess, alignment, restrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::column, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<const T> operator()() const {
      return std::span<const T>(params_.addr_, size_);
    }
    using NoParamReturnType = std::span<const T>;
    using ParamReturnType = const T&;
    SOA_HOST_DEVICE SOA_INLINE T const& operator()(size_type index) const { return params_.addr_[index]; }

  private:
    SoAConstParametersImpl<SoAColumnType::column, T> params_;
    const size_type size_ = 0;
  };

  // Scalar
  template <typename T, byte_size_type alignment, bool restrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::mutableAccess, alignment, restrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T& operator()() { return *params_.addr_; }
    using NoParamReturnType = T&;
    using ParamReturnType = void;
    SOA_HOST_DEVICE SOA_INLINE void operator()(size_type index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Const scalar
  template <typename T, byte_size_type alignment, bool restrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::constAccess, alignment, restrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE SOA_INLINE T const& operator()() const { return *params_.addr_; }
    using NoParamReturnType = T const&;
    using ParamReturnType = void;
    SOA_HOST_DEVICE SOA_INLINE void operator()(size_type index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAConstParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Eigen-type
  template <typename T, byte_size_type alignment, bool restrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::mutableAccess, alignment, restrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::eigen, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<typename T::Scalar> operator()() {
      return std::span<typename T::Scalar>(params_.addr_, size_);
    }
    using NoParamReturnType = std::span<typename T::Scalar>;
    using ParamReturnType = typename SoAValue<SoAColumnType::eigen, T, alignment, restrictQualify>::MapType;
    SOA_HOST_DEVICE SOA_INLINE ParamReturnType operator()(size_type index) {
      return SoAValue<SoAColumnType::eigen, T, alignment, restrictQualify>(index, params_)();
    }

  private:
    SoAParametersImpl<SoAColumnType::eigen, T> params_;
    size_type size_ = 0;
  };

  // Const Eigen-type
  template <typename T, byte_size_type alignment, bool restrictQualify>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::constAccess, alignment, restrictQualify> {
    SOA_HOST_DEVICE SOA_INLINE SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::eigen, T>& params,
                                                      size_type size)
        : params_(params), size_(size) {}
    SOA_HOST_DEVICE SOA_INLINE std::span<typename T::Scalar const> operator()() const {
      return std::span<typename T::Scalar const>(params_.addr_, size_);
    }
    using NoParamReturnType = std::span<typename T::Scalar const>;
    using ParamReturnType = typename SoAValue<SoAColumnType::eigen, T, alignment, restrictQualify>::CMapType;
    SOA_HOST_DEVICE SOA_INLINE ParamReturnType operator()(size_type index) const {
      return SoAConstValue<SoAColumnType::eigen, T, alignment, restrictQualify>(index, params_)();
    }

  private:
    SoAConstParametersImpl<SoAColumnType::eigen, T> params_;
    const size_type size_ = 0;
  };

  /* A helper template stager to avoid commas inside macros */
  template <typename T>
  struct SoAAccessors {
    template <auto columnType>
    struct ColumnType {
      template <auto accessType>
      struct AccessType {
        template <auto alignment>
        struct Alignment {
          template <auto restrictQualify>
          struct RestrictQualifier
              : public SoAColumnAccessorsImpl<T, columnType, accessType, alignment, restrictQualify> {
            using SoAColumnAccessorsImpl<T, columnType, accessType, alignment, restrictQualify>::SoAColumnAccessorsImpl;
          };
        };
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

namespace cms::soa::detail {

  template <RangeChecking::Mode M>
  struct IndexWithSourceLocation {
    static constexpr auto mode = M;

    SOA_HOST_DEVICE constexpr IndexWithSourceLocation(cms::soa::size_type value) noexcept : value_{value} {}

    cms::soa::size_type value_;
  };

  template <RangeChecking::Mode M>
    requires(M == RangeChecking::extended)
  struct IndexWithSourceLocation<M> {
    static constexpr auto mode = M;

    SOA_HOST_DEVICE constexpr IndexWithSourceLocation(
        cms::soa::size_type value, std::source_location location = std::source_location::current()) noexcept
        : value_{value}, location_{location} {}

    cms::soa::size_type value_;
    std::source_location location_;
  };

  template <RangeChecking::Mode M>
  [[noreturn]] void throwOutOfRangeError(const char* message,
                                         const IndexWithSourceLocation<M>& index,
                                         cms::soa::size_type range);

  // Helper function to check alignment of a pointer. Returns true if the pointer is not aligned to the specified alignment.
  [[noreturn]] void throwRuntimeError(const char* message);

  template <typename T>
  SOA_INLINE void checkAlignment(const T* addr, byte_size_type alignment, const char* message) {
    if (reinterpret_cast<uintptr_t>(addr) % alignment)
      throwRuntimeError(message);
  }

  template <typename ColumnType>
  struct PrintColumn;

  // Helper struct for streaming columns
  template <typename T>
  struct PrintColumn<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::scalar, T>> {
    void operator()(std::ostream& soa_impl_os,
                    std::string_view name,
                    cms::soa::byte_size_type& soa_impl_offset,
                    cms::soa::size_type,
                    cms::soa::byte_size_type alignment) {
      const auto size = sizeof(T);
      soa_impl_os << " Scalar " << name << " at offset " << soa_impl_offset << " has size " << size << " and padding "
                  << cms::soa::alignSize(size, alignment) - size << std::endl;
      soa_impl_offset += cms::soa::alignSize(size, alignment);
    }
  };

  template <typename T>
  struct PrintColumn<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::column, T>> {
    void operator()(std::ostream& soa_impl_os,
                    std::string_view name,
                    cms::soa::byte_size_type& soa_impl_offset,
                    cms::soa::size_type elements,
                    cms::soa::byte_size_type alignment) {
      const auto size = sizeof(T) * elements;
      soa_impl_os << " Column " << name << " at offset " << soa_impl_offset << " has size " << size << " and padding "
                  << cms::soa::alignSize(size, alignment) - size << std::endl;
      soa_impl_offset += cms::soa::alignSize(size, alignment);
    }
  };

  template <typename T>
  struct PrintColumn<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::eigen, T>> {
    void operator()(std::ostream& soa_impl_os,
                    std::string_view name,
                    cms::soa::byte_size_type& soa_impl_offset,
                    cms::soa::size_type elements,
                    cms::soa::byte_size_type alignment) {
      const auto size = elements * sizeof(typename T::Scalar);
      soa_impl_os << " Eigen value " << name << " at offset " << soa_impl_offset << " has dimension "
                  << "(" << T::RowsAtCompileTime << " x " << T::ColsAtCompileTime << ")"
                  << " and per column size " << size << " and padding " << cms::soa::alignSize(size, alignment) - size
                  << std::endl;
      soa_impl_offset += cms::soa::alignSize(size, alignment) * T::RowsAtCompileTime * T::ColsAtCompileTime;
    }
  };

  // Helper struct for computing the pitch of each column
  template <typename ColumnType>
  struct ComputePitch;

  template <typename T>
  struct ComputePitch<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::scalar, T>> {
    SOA_HOST_DEVICE constexpr cms::soa::byte_size_type operator()(cms::soa::size_type,
                                                                  cms::soa::byte_size_type alignment) const {
      return cms::soa::alignSize(sizeof(T), alignment);
    }
  };

  template <typename T>
  struct ComputePitch<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::column, T>> {
    SOA_HOST_DEVICE constexpr cms::soa::byte_size_type operator()(cms::soa::size_type elements,
                                                                  cms::soa::byte_size_type alignment) const {
      return cms::soa::alignSize(elements * sizeof(T), alignment);
    }
  };

  template <typename T>
  struct ComputePitch<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::eigen, T>> {
    SOA_HOST_DEVICE constexpr cms::soa::byte_size_type operator()(cms::soa::size_type elements,
                                                                  cms::soa::byte_size_type alignment) const {
      return cms::soa::alignSize(elements * sizeof(typename T::Scalar), alignment) * T::RowsAtCompileTime *
             T::ColsAtCompileTime;
    }
  };

  // Helper type trait for obtaining the underlying type of an enum, or the type itself if it's not an enum
  template <typename T>
  struct EnumTraits {
    using type = T;
    using value_type = T;
  };

  template <typename T>
    requires std::is_enum_v<T>
  struct EnumTraits<T> {
    using type = T;
    using value_type = std::underlying_type_t<T>;
  };

  // Helper type trait for obtaining a span type for a column
  template <typename ColumnType>
  struct GetSpanType;

  template <typename T>
  struct GetSpanType<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::scalar, T>> {
    using type = std::span<T, 1>;
  };

  template <typename T>
  struct GetSpanType<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::column, T>> {
    using type = std::span<T>;
  };

  template <typename T>
  struct GetSpanType<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::eigen, T>> {
    using type = std::span<typename T::Scalar>;
  };

  template <typename ColumnType>
  using SpanType = GetSpanType<ColumnType>::type;

  // Helper type trait for obtaining a const-span type for a column
  template <typename ColumnType>
  struct GetConstSpanType;

  template <typename T>
  struct GetConstSpanType<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::scalar, T>> {
    using type = std::span<std::add_const_t<T>, 1>;
  };

  template <typename T>
  struct GetConstSpanType<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::column, T>> {
    using type = std::span<std::add_const_t<T>>;
  };

  template <typename T>
  struct GetConstSpanType<cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::eigen, T>> {
    using type = std::span<std::add_const_t<typename T::Scalar>>;
  };

  template <typename ColumnType>
  using ConstSpanType = GetConstSpanType<ColumnType>::type;

  // Helper functions for constructing a span from a column
  template <typename T>
  auto getSpanToColumn(const cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::scalar, T>& column,
                       cms::soa::size_type,
                       cms::soa::byte_size_type alignment) {
    return std::span(column.addr_, 1);
  }

  template <typename T>
  auto getSpanToColumn(const cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::column, T>& column,
                       cms::soa::size_type elements,
                       cms::soa::byte_size_type alignment) {
    return std::span(column.addr_, elements);
  }

  template <typename T>
  auto getSpanToColumn(const cms::soa::SoAParametersImpl<cms::soa::SoAColumnType::eigen, T>& column,
                       cms::soa::size_type elements,
                       cms::soa::byte_size_type alignment) {
    return std::span(column.addr_,
                     cms::soa::alignSize(elements * sizeof(typename T::Scalar), alignment) * T::RowsAtCompileTime *
                         T::ColsAtCompileTime / sizeof(typename T::Scalar));
  }

  // Helper function for extracting the number of blocks of a layout. Falls back to 1 if the layout does not define a static member blocksNumber.
  template <typename T>
  constexpr size_type nBlocks() {
    if constexpr (requires { T::blocksNumber; })
      return T::blocksNumber;
    else
      return static_cast<size_type>(1);
  }

  // Case 1: type has blocksNumber → returns a sub-array
  template <typename T, size_type N>
    requires requires { T::blocksNumber; }
  [[nodiscard]] constexpr std::array<size_type, T::blocksNumber> extractSegment(const std::array<size_type, N>& sizes,
                                                                                size_type offset) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
      return std::array<size_type, T::blocksNumber>{sizes[offset + I]...};
    }(std::make_index_sequence<T::blocksNumber>{});
  }

  // Case 2: fallback (single block) → returns a scalar
  template <typename T, size_type N>
    requires(!requires { T::blocksNumber; })
  [[nodiscard]] constexpr size_type extractSegment(const std::array<size_type, N>& sizes, size_type offset) {
    return sizes[offset];
  }

}  // namespace cms::soa::detail

#endif  // DataFormats_SoATemplate_interface_SoACommon_h
