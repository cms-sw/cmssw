/*
 * Definitions of SoA common parameters for SoA class generators
 */

#ifndef DataStructures_SoACommon_h
#define DataStructures_SoACommon_h

#include "boost/preprocessor.hpp"
#include <Eigen/Core>

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_DEVICE_ONLY __device__
#define SOA_HOST_DEVICE __host__ __device__
#define SOA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define SOA_DEVICE_RESTRICT __restrict__
#else
#define SOA_HOST_ONLY
#define SOA_DEVICE_ONLY
#define SOA_HOST_DEVICE
#define SOA_HOST_DEVICE_INLINE inline
#define SOA_DEVICE_RESTRICT
#endif

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
// Read a pointer content via read-only (non coherent) cache.
#define LOAD_INCOHERENT(A) __ldg(A)
#define LOAD_STREAMED(A) __ldcs(A)
#define STORE_STREAMED(A, V) __stcs(A, V)
#else
#define LOAD_INCOHERENT(A) *(A)
#define LOAD_STREAMED(A) *(A)
#define STORE_STREAMED(A, V) *(A) = (V)
#endif

// compile-time sized SoA

// Helper template managing the value within it column
// The optional compile time alignment parameter enables informing the
// compiler of alignment (enforced by caller).
template <typename T, size_t ALIGNMENT>
class SoAValue {
public:
  SOA_HOST_DEVICE_INLINE SoAValue(size_t i, T* col) : idx_(i), col_(col) {}
  /* SOA_HOST_DEVICE_INLINE operator T&() { return col_[idx_]; } */
  SOA_HOST_DEVICE_INLINE T& operator()() { return alignedCol()[idx_]; }
  SOA_HOST_DEVICE_INLINE T operator()() const { return *(alignedCol() + idx_); }
  SOA_HOST_DEVICE_INLINE T* operator&() { return &alignedCol()[idx_]; }
  SOA_HOST_DEVICE_INLINE const T* operator&() const { return &alignedCol()[idx_]; }
  template <typename T2>
  SOA_HOST_DEVICE_INLINE T& operator=(const T2& v) {
    return alignedCol()[idx_] = v;
  }
  typedef T valueType;
  static constexpr auto valueSize = sizeof(T);

private:
  SOA_HOST_DEVICE_INLINE T* alignedCol() const {
    if constexpr (ALIGNMENT) {
      return reinterpret_cast<T*>(__builtin_assume_aligned(col_, ALIGNMENT));
    } else {
      return col_;
    }
  }
  size_t idx_;
  T* col_;
};

// Helper template managing the value within it column
template <typename T, size_t ALIGNMENT>
class SoAConstValue {
public:
  SOA_HOST_DEVICE_INLINE SoAConstValue(size_t i, const T* col) : idx_(i), col_(col) {}
  /* SOA_HOST_DEVICE_INLINE operator T&() { return col_[idx_]; } */
  SOA_HOST_DEVICE_INLINE T operator()() const { return *(alignedCol() + idx_); }
  SOA_HOST_DEVICE_INLINE const T* operator&() const { return &alignedCol()[idx_]; }
  typedef T valueType;
  static constexpr auto valueSize = sizeof(T);

private:
  SOA_HOST_DEVICE_INLINE const T* alignedCol() const {
    if constexpr (ALIGNMENT) {
      return __builtin_assume_aligned(col_, ALIGNMENT);
    } else {
      return col_;
    }
  }
  size_t idx_;
  const T* col_;
};

// Helper template managing the value within it column
template <class C, size_t ALIGNMENT>
class SoAEigenValue {
public:
  typedef C Type;
  typedef Eigen::Map<C, 0, Eigen::InnerStride<Eigen::Dynamic>> MapType;
  typedef Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>> CMapType;
  SOA_HOST_DEVICE_INLINE SoAEigenValue(size_t i, typename C::Scalar* col, size_t stride)
      : val_(col + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
        crCol_(col),
        cVal_(crCol_ + i, C::RowsAtCompileTime, C::ColsAtCompileTime, Eigen::InnerStride<Eigen::Dynamic>(stride)),
        stride_(stride) {}
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
  SOA_HOST_DEVICE_INLINE size_t stride() { return stride_; }

private:
  MapType val_;
  const typename C::Scalar* __restrict__ crCol_;
  CMapType cVal_;
  size_t stride_;
};

// Helper template to avoid commas in macro
template <class C>
struct EigenConstMapMaker {
  typedef Eigen::Map<const C, 0, Eigen::InnerStride<Eigen::Dynamic>> Type;
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

// Helper function to compute aligned size
inline size_t alignSize(size_t size, size_t alignment = 128) {
  if (size)
    return ((size - 1) / alignment + 1) * alignment;
  else
    return 0;
}

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1
#define _VALUE_TYPE_EIGEN_COLUMN 2

enum class SoAColumnType { scalar = _VALUE_TYPE_SCALAR, column = _VALUE_TYPE_COLUMN, eigen = _VALUE_TYPE_EIGEN_COLUMN };

#define SoA_scalar(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME)
#define SoA_column(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME)
#define SoA_eigenColumn(TYPE, NAME) (_VALUE_TYPE_EIGEN_COLUMN, TYPE, NAME)

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

/* Enum parameters allowing templated control of store/view behaviors */
/* Alignement enforcement verifies every column is aligned, and 
 * hints the compiler that it can expect column pointers to be aligned */
enum class AlignmentEnforcement : bool { Relaxed, Enforced };

#endif  // ndef DataStructures_SoACommon_h
