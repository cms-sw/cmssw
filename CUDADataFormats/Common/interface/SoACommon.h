/*
 * Definitions of SoA common parameters for SoA class generators
 */

#ifndef DataStructures_SoACommon_h
#define DataStructures_SoACommon_h

#include "boost/preprocessor.hpp"
#include <cstdint>
#include <cassert>
#include <ostream>

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_DEVICE_ONLY __device__
#define SOA_HOST_DEVICE __host__ __device__
#define SOA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define SOA_HOST_ONLY
#define SOA_DEVICE_ONLY
#define SOA_HOST_DEVICE
#define SOA_HOST_DEVICE_INLINE inline
#endif

// Exception throwing (or willful crash in kernels)
#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define SOA_THROW_OUT_OF_RANGE(A) \
  {                               \
    printf(A "\n");               \
    *((char*)nullptr) = 0;        \
  }
#else
#define SOA_THROW_OUT_OF_RANGE(A) \
  { throw std::out_of_range(A); }
#endif

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1
#define _VALUE_TYPE_EIGEN_COLUMN 2

namespace cms::soa {

  enum class SoAColumnType {
    scalar = _VALUE_TYPE_SCALAR,
    column = _VALUE_TYPE_COLUMN,
    eigen = _VALUE_TYPE_EIGEN_COLUMN
  };
  enum class RestrictQualify : bool { Enabled, Disabled, Default = Disabled };

  enum class RangeChecking : bool { Enabled, Disabled, Default = Disabled };

  template <typename T, RestrictQualify RESTRICT_QUALIFY>
  struct add_restrict {};

  template <typename T>
  struct add_restrict<T, RestrictQualify::Enabled> {
    typedef T Value;
    typedef T* __restrict__ Pointer;
    typedef T& __restrict__ Reference;
    typedef const T ConstValue;
    typedef const T* __restrict__ PointerToConst;
    typedef const T& __restrict__ ReferenceToConst;
  };

  template <typename T>
  struct add_restrict<T, RestrictQualify::Disabled> {
    typedef T Value;
    typedef T* Pointer;
    typedef T& Reference;
    typedef const T ConstValue;
    typedef const T* PointerToConst;
    typedef const T& ReferenceToConst;
  };
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAParametersImpl;

  // Templated parameter sets for scalar columns and Eigen columns
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAConstParametersImpl {
    static const SoAColumnType columnType = COLUMN_TYPE;
    typedef T ValueType;
    typedef const ValueType* TupleOrPointerType;
    const ValueType* addr_ = nullptr;
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const ValueType* addr) : addr_(addr) {}
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const SoAConstParametersImpl& o) { addr_ = o.addr_; }
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const SoAParametersImpl<columnType, ValueType>& o) {
      addr_ = o.addr_;
    }
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl() {}
    static bool checkAlignement(ValueType* addr, size_t byteAlignment) {
      return reinterpret_cast<intptr_t>(addr) % byteAlignment;
    }
  };

  template <typename T>
  struct SoAConstParametersImpl<SoAColumnType::eigen, T> {
    static const SoAColumnType columnType = SoAColumnType::eigen;
    typedef T ValueType;
    typedef typename T::Scalar ScalarType;
    typedef std::tuple<ScalarType*, size_t> TupleOrPointerType;
    const ScalarType* addr_ = nullptr;
    size_t stride_ = 0;
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const ScalarType* addr, size_t stride)
        : addr_(addr), stride_(stride) {}
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const TupleOrPointerType tuple)
        : addr_(std::get<0>(tuple)), stride_(std::get<1>(tuple)) {}
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const ScalarType* addr) : addr_(addr) {}
    // Trick setter + return self-reference allowing commat-free 2-stage construction in macro contexts (in combination with the
    // addr-only constructor.
    SoAConstParametersImpl& setStride(size_t stride) {
      stride_ = stride;
      return *this;
    }
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const SoAConstParametersImpl& o) {
      addr_ = o.addr_;
      stride_ = o.stride_;
    }
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl(const SoAParametersImpl<columnType, ValueType>& o) {
      addr_ = o.addr_;
      stride_ = o.stride_;
    }
    SOA_HOST_DEVICE_INLINE SoAConstParametersImpl() {}
    static bool checkAlignement(const TupleOrPointerType tuple, size_t byteAlignment) {
      const auto& [addr, stride] = tuple;
      return reinterpret_cast<intptr_t>(addr) % byteAlignment;
    }
  };

  // Matryoshka template to avoiding commas in macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAConstParameters_ColumnType {
    template <typename T>
    struct DataType : public SoAConstParametersImpl<COLUMN_TYPE, T> {
      using SoAConstParametersImpl<COLUMN_TYPE, T>::SoAConstParametersImpl;
    };
  };

  // Templated parameter sets for scalar columns and Eigen columns
  template <SoAColumnType COLUMN_TYPE, typename T>
  struct SoAParametersImpl {
    static const SoAColumnType columnType = COLUMN_TYPE;
    typedef T ValueType;
    typedef const ValueType* TupleOrPointerType;
    typedef SoAConstParametersImpl<columnType, ValueType> ConstType;
    friend ConstType;
    ValueType* addr_ = nullptr;
    SOA_HOST_DEVICE_INLINE SoAParametersImpl(ValueType* addr) : addr_(addr) {}
    SOA_HOST_DEVICE_INLINE SoAParametersImpl() {}
    static bool checkAlignement(ValueType* addr, size_t byteAlignment) {
      return reinterpret_cast<intptr_t>(addr) % byteAlignment;
    }
  };

  template <typename T>
  struct SoAParametersImpl<SoAColumnType::eigen, T> {
    static const SoAColumnType columnType = SoAColumnType::eigen;
    typedef T ValueType;
    typedef SoAConstParametersImpl<columnType, ValueType> ConstType;
    friend ConstType;
    typedef typename T::Scalar ScalarType;
    typedef std::tuple<ScalarType*, size_t> TupleOrPointerType;
    ScalarType* addr_ = nullptr;
    size_t stride_ = 0;
    SOA_HOST_DEVICE_INLINE SoAParametersImpl(ScalarType* addr, size_t stride)
        : addr_(addr), stride_(stride) {}
    SOA_HOST_DEVICE_INLINE SoAParametersImpl(const TupleOrPointerType tuple)
        : addr_(std::get<0>(tuple)), stride_(std::get<1>(tuple)) {}
    SOA_HOST_DEVICE_INLINE SoAParametersImpl() {}
    SOA_HOST_DEVICE_INLINE SoAParametersImpl(ScalarType* addr) : addr_(addr) {}
    // Trick setter + return self-reference allowing commat-free 2-stage construction in macro contexts (in combination with the
    // addr-only constructor.
    SoAParametersImpl& setStride(size_t stride) {
      stride_ = stride;
      return *this;
    }
    static bool checkAlignement(const TupleOrPointerType tuple, size_t byteAlignment) {
      const auto& [addr, stride] = tuple;
      return reinterpret_cast<intptr_t>(addr) % byteAlignment;
    }
  };

  // Matryoshka template to avoiding commas in macros
  template <SoAColumnType COLUMN_TYPE>
  struct SoAParameters_ColumnType {
    template <typename T>
    struct DataType : public SoAParametersImpl<COLUMN_TYPE, T> {
      using SoAParametersImpl<COLUMN_TYPE, T>::SoAParametersImpl;
    };
  };

  // Helper template managing the value within it column
  // The optional compile time alignment parameter enables informing the
  // compiler of alignment (enforced by caller).
  template <SoAColumnType COLUMN_TYPE,
            typename T,
            size_t ALIGNMENT,
            RestrictQualify RESTRICT_QUALIFY = RestrictQualify::Disabled>
  class SoAValue {
    // Eigen is implemented in a specialization
    static_assert(COLUMN_TYPE != SoAColumnType::eigen);

  public:
    typedef add_restrict<T, RESTRICT_QUALIFY> Restr;
    typedef typename Restr::Value Val;
    typedef typename Restr::Pointer Ptr;
    typedef typename Restr::Reference Ref;
    typedef typename Restr::PointerToConst PtrToConst;
    typedef typename Restr::ReferenceToConst RefToConst;
    SOA_HOST_DEVICE_INLINE SoAValue(size_t i, T* col) : idx_(i), col_(col) {}
    SOA_HOST_DEVICE_INLINE SoAValue(size_t i, SoAParametersImpl<COLUMN_TYPE, T> params)
        : idx_(i), col_(params.addr_) {}
    /* SOA_HOST_DEVICE_INLINE operator T&() { return col_[idx_]; } */
    SOA_HOST_DEVICE_INLINE Ref operator()() {
      // Ptr type will add the restrict qualifyer if needed
      Ptr col = alignedCol();
      return col[idx_];
    }
    SOA_HOST_DEVICE_INLINE RefToConst operator()() const {
      // PtrToConst type will add the restrict qualifyer if needed
      PtrToConst col = alignedCol();
      return col[idx_];
    }
    SOA_HOST_DEVICE_INLINE Ptr operator&() { return &alignedCol()[idx_]; }
    SOA_HOST_DEVICE_INLINE PtrToConst operator&() const { return &alignedCol()[idx_]; }
    template <typename T2>
    SOA_HOST_DEVICE_INLINE Ref operator=(const T2& v) {
      return alignedCol()[idx_] = v;
    }
    typedef Val valueType;
    static constexpr auto valueSize = sizeof(T);

  private:
    SOA_HOST_DEVICE_INLINE Ptr alignedCol() const {
      if constexpr (ALIGNMENT) {
        return reinterpret_cast<Ptr>(__builtin_assume_aligned(col_, ALIGNMENT));
      }
      return reinterpret_cast<Ptr>(col_);
    }
    size_t idx_;
    T* col_;
  };

  // Helper template managing the value within it column
  // TODO Create a const variant to avoid leaking mutable access.
#ifdef EIGEN_WORLD_VERSION
  template <class C, size_t ALIGNMENT, RestrictQualify RESTRICT_QUALIFY>
  class SoAValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    typedef C Type;
    typedef Eigen::Map<C, 0, Eigen::InnerStride<Eigen::Dynamic>> MapType;
    typedef Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>> CMapType;
    typedef add_restrict<typename C::Scalar, RESTRICT_QUALIFY> Restr;
    typedef typename Restr::Value Val;
    typedef typename Restr::Pointer Ptr;
    typedef typename Restr::Reference Ref;
    typedef typename Restr::PointerToConst PtrToConst;
    typedef typename Restr::ReferenceToConst RefToConst;
    SOA_HOST_DEVICE_INLINE SoAValue(size_t i, typename C::Scalar* col, size_t stride)
        : val_(col + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          stride_(stride) {}
    SOA_HOST_DEVICE_INLINE SoAValue(size_t i, SoAParametersImpl<SoAColumnType::eigen, C> params)
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
    SOA_HOST_DEVICE_INLINE MapType& operator()() { return val_; }
    SOA_HOST_DEVICE_INLINE const CMapType& operator()() const { return cVal_; }
    SOA_HOST_DEVICE_INLINE operator C() { return val_; }
    SOA_HOST_DEVICE_INLINE operator const C() const { return cVal_; }
    SOA_HOST_DEVICE_INLINE C* operator&() { return &val_; }
    SOA_HOST_DEVICE_INLINE const C* operator&() const { return &cVal_; }
    template <class C2>
    SOA_HOST_DEVICE_INLINE MapType& operator=(const C2& v) {
      return val_ = v;
    }
    typedef typename C::Scalar ValueType;
    static constexpr auto valueSize = sizeof(C::Scalar);
    SOA_HOST_DEVICE_INLINE size_t stride() const { return stride_; }

  private:
    MapType val_;
    const Ptr crCol_;
    CMapType cVal_;
    size_t stride_;
  };
#else
  template <class C, size_t ALIGNMENT, RestrictQualify RESTRICT_QUALIFY>
  class SoAValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
    // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif
  // Helper template managing the value within it column
  template <SoAColumnType COLUMN_TYPE,
            typename T,
            size_t ALIGNMENT,
            RestrictQualify RESTRICT_QUALIFY = RestrictQualify::Disabled>
  class SoAConstValue {
    // Eigen is implemented in a specialization
    static_assert(COLUMN_TYPE != SoAColumnType::eigen);

  public:
    typedef add_restrict<T, RESTRICT_QUALIFY> Restr;
    typedef typename Restr::Value Val;
    typedef typename Restr::Pointer Ptr;
    typedef typename Restr::Reference Ref;
    typedef typename Restr::PointerToConst PtrToConst;
    typedef typename Restr::ReferenceToConst RefToConst;
    typedef SoAParametersImpl<COLUMN_TYPE, T> Params;
    typedef SoAConstParametersImpl<COLUMN_TYPE, T> ConstParams;
    SOA_HOST_DEVICE_INLINE SoAConstValue(size_t i, const T* col) : idx_(i), col_(col) {}
    SOA_HOST_DEVICE_INLINE SoAConstValue(size_t i, SoAParametersImpl<COLUMN_TYPE, T> params)
        : idx_(i), col_(params.addr_) {}
    SOA_HOST_DEVICE_INLINE SoAConstValue(size_t i, SoAConstParametersImpl<COLUMN_TYPE, T> params)
        : idx_(i), col_(params.addr_) {}
    /* SOA_HOST_DEVICE_INLINE operator T&() { return col_[idx_]; } */
    SOA_HOST_DEVICE_INLINE RefToConst operator()() const {
      // Ptr type will add the restrict qualifyer if needed
      PtrToConst col = alignedCol();
      return col[idx_];
    }
    SOA_HOST_DEVICE_INLINE const T* operator&() const { return &alignedCol()[idx_]; }
    typedef T valueType;
    static constexpr auto valueSize = sizeof(T);

  private:
    SOA_HOST_DEVICE_INLINE PtrToConst alignedCol() const {
      if constexpr (ALIGNMENT) {
        return reinterpret_cast<PtrToConst>(__builtin_assume_aligned(col_, ALIGNMENT));
      }
      return reinterpret_cast<PtrToConst>(col_);
    }
    size_t idx_;
    const T* col_;
  };

#ifdef EIGEN_WORLD_VERSION
  // Helper template managing the value within it column
  // TODO Create a const variant to avoid leaking mutable access.
  template <class C, size_t ALIGNMENT, RestrictQualify RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
  public:
    typedef C Type;
    typedef Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>> CMapType;
    typedef CMapType& RefToConst;
    typedef SoAConstParametersImpl<SoAColumnType::eigen, C> ConstParams;
    SOA_HOST_DEVICE_INLINE SoAConstValue(size_t i, typename C::Scalar* col, size_t stride)
        : crCol_(col),
          cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
          stride_(stride) {}
    SOA_HOST_DEVICE_INLINE SoAConstValue(size_t i, SoAConstParametersImpl<SoAColumnType::eigen, C> params)
        : crCol_(params.addr_),
          cVal_(crCol_ + i,
                C::RowsAtCompileTime,
                C::ColsAtCompileTime,
                Eigen::InnerStride<Eigen::Dynamic>(params.stride_)),
          stride_(params.stride_) {}
    SOA_HOST_DEVICE_INLINE const CMapType& operator()() const { return cVal_; }
    SOA_HOST_DEVICE_INLINE operator const C() const { return cVal_; }
    SOA_HOST_DEVICE_INLINE const C* operator&() const { return &cVal_; }
    typedef typename C::Scalar ValueType;
    static constexpr auto valueSize = sizeof(C::Scalar);
    SOA_HOST_DEVICE_INLINE size_t stride() const { return stride_; }

  private:
    const typename C::Scalar* __restrict__ crCol_;
    CMapType cVal_;
    size_t stride_;
  };
#else
  template <class C, size_t ALIGNMENT, RestrictQualify RESTRICT_QUALIFY>
  class SoAConstValue<SoAColumnType::eigen, C, ALIGNMENT, RESTRICT_QUALIFY> {
    // Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.
    static_assert(!sizeof(C),
                  "Eigen/Core should be pre-included before the SoA headers to enable support for Eigen columns.");
  };
#endif

  // Helper template to avoid commas in macro
#ifdef EIGEN_WORLD_VERSION
  template <class C>
  struct EigenConstMapMaker {
    typedef Eigen::Map<const C, Eigen::AlignmentType::Unaligned, Eigen::InnerStride<Eigen::Dynamic>> Type;
    class DataHolder {
    public:
      DataHolder(const typename C::Scalar* data) : data_(data) {}
      EigenConstMapMaker::Type withStride(size_t stride) {
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
  inline size_t alignSize(size_t size, size_t alignment = 128) {
    if (size)
      return ((size - 1) / alignment + 1) * alignment;
    else
      return 0;
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

  // Todo: add alignment support.
  // Sfinae based const/non const variants.
  // Column
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::mutableAccess> {
    //SOA_HOST_DEVICE_INLINE SoAColumnAccessorsImpl(T* baseAddress) : baseAddress_(baseAddress) {}
    SOA_HOST_DEVICE_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::column, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE_INLINE T* operator()() { return params_.addr_; }
    typedef T* NoParamReturnType;
    SOA_HOST_DEVICE_INLINE T& operator()(size_t index) { return params_.addr_[index]; }

  private:
    SoAParametersImpl<SoAColumnType::column, T> params_;
  };

  // Const column
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::column, SoAAccessType::constAccess> {
    SOA_HOST_DEVICE_INLINE
    SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::column, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE_INLINE const T* operator()() const { return params_.addr_; }
    typedef T* NoParamReturnType;
    SOA_HOST_DEVICE_INLINE T operator()(size_t index) const { return params_.addr_[index]; }

  private:
    SoAConstParametersImpl<SoAColumnType::column, T> params_;
  };

  // Scalar
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::mutableAccess> {
    SOA_HOST_DEVICE_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE_INLINE T& operator()() { return *params_.addr_; }
    typedef T& NoParamReturnType;
    SOA_HOST_DEVICE_INLINE void operator()(size_t index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAParametersImpl<SoAColumnType::scalar, T> params_;
  };

  // Const scalar
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::scalar, SoAAccessType::constAccess> {
    SOA_HOST_DEVICE_INLINE
    SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::scalar, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE_INLINE T operator()() const { return *params_.addr_; }
    typedef T NoParamReturnType;
    SOA_HOST_DEVICE_INLINE void operator()(size_t index) const {
      assert(false && "Indexed access impossible for SoA scalars.");
    }

  private:
    SoAConstParametersImpl<SoAColumnType::scalar, T> params_;
  };

  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::mutableAccess> {
    //SOA_HOST_DEVICE_INLINE SoAColumnAccessorsImpl(T* baseAddress) : baseAddress_(baseAddress) {}
    SOA_HOST_DEVICE_INLINE SoAColumnAccessorsImpl(const SoAParametersImpl<SoAColumnType::eigen, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE_INLINE typename T::Scalar* operator()() { return params_.addr_; }
    typedef typename T::Scalar* NoParamReturnType;
    //SOA_HOST_DEVICE_INLINE T& operator()(size_t index) { return params_.addr_[index]; }

  private:
    SoAParametersImpl<SoAColumnType::eigen, T> params_;
  };

  // Const column
  template <typename T>
  struct SoAColumnAccessorsImpl<T, SoAColumnType::eigen, SoAAccessType::constAccess> {
    SOA_HOST_DEVICE_INLINE
    SoAColumnAccessorsImpl(const SoAConstParametersImpl<SoAColumnType::eigen, T>& params)
        : params_(params) {}
    SOA_HOST_DEVICE_INLINE const typename T::Scalar* operator()() const { return params_.addr_; }
    typedef typename T::Scalar* NoParamReturnType;
    //SOA_HOST_DEVICE_INLINE T operator()(size_t index) const { return params_.addr_[index]; }

  private:
    SoAConstParametersImpl<SoAColumnType::eigen, T> params_;
  };
  
  /* A helper template stager avoiding comma in macros */
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
  /* Alignement enforcement verifies every column is aligned, and 
 * hints the compiler that it can expect column pointers to be aligned */
  enum class AlignmentEnforcement : bool { Relaxed, Enforced };

  struct CacheLineSize {
    static constexpr size_t NvidiaGPU = 128;
    static constexpr size_t IntelCPU = 64;
    static constexpr size_t AMDCPU = 64;
    static constexpr size_t ARMCPU = 64;
    static constexpr size_t defaultSize = NvidiaGPU;
  };

  // An empty shell class to restrict the scope of tempalted operator<<(ostream, soa).
  struct BaseLayout {};
}  // namespace cms::soa

// Small wrapper for stream insertion of SoA printing
template <typename SOA,
          typename SOACHECKED = typename std::enable_if<std::is_base_of<cms::soa::BaseLayout, SOA>::value, SOA>::type>
SOA_HOST_ONLY std::ostream& operator<<(std::ostream& os, const SOA& soa) {
  soa.toStream(os);
  return os;
}
#endif  // ndef DataStructures_SoACommon_h
