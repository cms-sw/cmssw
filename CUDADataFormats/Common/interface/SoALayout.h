/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#ifndef DataStructures_SoALayout_h
#define DataStructures_SoALayout_h

#include "SoACommon.h"

#include <iostream>
#include <cassert>

/* dump SoA fields information; these should expand to, for columns:
 * Example:
 * GENERATE_SOA_LAYOUT(SoA,
 *   // predefined static scalars
 *   // size_t size;
 *   // size_t alignment;
 *
 *   // columns: one value per element
 *   SOA_COLUMN(double, x),
 *   SOA_COLUMN(double, y),
 *   SOA_COLUMN(double, z),
 *   SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
 *   SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
 *   SOA_EIGEN_COLUMN(Eigen::Vector3d, r),
 *   SOA_COLUMN(uint16_t, colour),
 *   SOA_COLUMN(int32_t, value),
 *   SOA_COLUMN(double *, py),
 *   SOA_COLUMN(uint32_t, count),
 *   SOA_COLUMN(uint32_t, anotherCount),
 *
 *   // scalars: one value for the whole structure
 *   SOA_SCALAR(const char *, description),
 *   SOA_SCALAR(uint32_t, someNumber)
 * );
 * 
 * dumps as:
 * SoA(32, 64):
 *   sizeof(SoA): 152
 *  Column x_ at offset 0 has size 256 and padding 0
 *  Column y_ at offset 256 has size 256 and padding 0
 *  Column z_ at offset 512 has size 256 and padding 0
 *  Eigen value a_ at offset 768 has dimension (3 x 1) and per column size 256 and padding 0
 *  Eigen value b_ at offset 1536 has dimension (3 x 1) and per column size 256 and padding 0
 *  Eigen value r_ at offset 2304 has dimension (3 x 1) and per column size 256 and padding 0
 *  Column colour_ at offset 3072 has size 64 and padding 0
 *  Column value_ at offset 3136 has size 128 and padding 0
 *  Column py_ at offset 3264 has size 256 and padding 0
 *  Column count_ at offset 3520 has size 128 and padding 0
 *  Column anotherCount_ at offset 3648 has size 128 and padding 0
 *  Scalar description_ at offset 3776 has size 8 and padding 56
 *  Scalar someNumber_ at offset 3840 has size 4 and padding 60
 * Final offset = 3904 computeDataSize(...): 3904
 *
 */

// clang-format off
#define _DECLARE_SOA_STREAM_INFO_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                         \
  _SWITCH_ON_TYPE(                                                                                                        \
      VALUE_TYPE,                                                                                                         \
      /* Dump scalar */                                                                                                   \
      os << " Scalar " BOOST_PP_STRINGIZE(NAME) " at offset " << offset << " has size " << sizeof(CPP_TYPE)               \
         << " and padding " << ((sizeof(CPP_TYPE) - 1) / byteAlignment + 1) * byteAlignment - sizeof(CPP_TYPE)            \
         << std::endl;                                                                                                    \
      offset += ((sizeof(CPP_TYPE) - 1) / byteAlignment + 1) * byteAlignment;                                             \
      , /* Dump column */                                                                                                 \
      os << " Column " BOOST_PP_STRINGIZE(NAME) " at offset " << offset << " has size " << sizeof(CPP_TYPE) * nElements_  \
         << " and padding "                                                                                               \
         << (((nElements_ * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment - (sizeof(CPP_TYPE) * nElements_) \
         << std::endl;                                                                                                    \
      offset += (((nElements_ * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                              \
      , /* Dump Eigen column */                                                                                           \
      os << " Eigen value " BOOST_PP_STRINGIZE(NAME) " at offset " << offset << " has dimension ("                        \
         << CPP_TYPE::RowsAtCompileTime << " x " << CPP_TYPE::ColsAtCompileTime                                           \
         << ")"                                                                                                           \
         << " and per column size "                                                                                       \
         << sizeof(CPP_TYPE::Scalar) * nElements_                                                                         \
         << " and padding "                                                                                               \
         << (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment -                         \
                (sizeof(CPP_TYPE::Scalar) * nElements_)                                                                   \
         << std::endl;                                                                                                    \
      offset += (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment *                     \
                                 CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;)
// clang-format on

#define _DECLARE_SOA_STREAM_INFO(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_STREAM_INFO_IMPL TYPE_NAME)

/**
 * SoAMetadata member computing column pitch
 */
// clang-format off
#define _DEFINE_METADATA_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                    \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
      /* Scalar */                                                                                                   \
      size_t BOOST_PP_CAT(NAME, Pitch()) const {                                                                     \
        return (((sizeof(CPP_TYPE) - 1) / ParentClass::byteAlignment) + 1) * ParentClass::byteAlignment;             \
      }                                                                                                              \
      typedef CPP_TYPE BOOST_PP_CAT(TypeOf_, NAME);                                                                  \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::scalar;  \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      CPP_TYPE const* BOOST_PP_CAT(addressOf_, NAME)() const {                                                       \
        return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                      \
      }                                                                                                              \
      typedef cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::scalar>::DataType<CPP_TYPE>                \
        BOOST_PP_CAT(ParametersTypeOf_, NAME);                                                                       \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                              \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (parent_.BOOST_PP_CAT(NAME, _));                               \
      }                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      CPP_TYPE* BOOST_PP_CAT(addressOf_, NAME)() {                                                                   \
        return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                      \
      },                                                                                                             \
      /* Column */                                                                                                   \
      typedef cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<CPP_TYPE>                \
        BOOST_PP_CAT(ParametersTypeOf_, NAME);                                                                       \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                              \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (parent_.BOOST_PP_CAT(NAME, _));                               \
      }                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      CPP_TYPE const* BOOST_PP_CAT(addressOf_, NAME)() const {                                                       \
        return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                      \
      }                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      CPP_TYPE* BOOST_PP_CAT(addressOf_, NAME)() {                                                                   \
        return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                      \
      }                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      size_t BOOST_PP_CAT(NAME, Pitch()) const {                                                                     \
        return (((parent_.nElements_ * sizeof(CPP_TYPE) - 1) / ParentClass::byteAlignment) + 1) *                    \
                   ParentClass::byteAlignment;                                                                       \
      }                                                                                                              \
      typedef CPP_TYPE BOOST_PP_CAT(TypeOf_, NAME);                                                                  \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::column;, \
      /* Eigen column */                                                                                             \
      typedef cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::eigen>::DataType<CPP_TYPE>                 \
        BOOST_PP_CAT(ParametersTypeOf_, NAME);                                                                       \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                              \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (                                                              \
         parent_.BOOST_PP_CAT(NAME, _),                                                                              \
         parent_.BOOST_PP_CAT(NAME, Stride_));                                                                       \
      }                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      size_t BOOST_PP_CAT(NAME, Pitch()) const {                                                                     \
        return (((parent_.nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / ParentClass::byteAlignment) + 1) *            \
               ParentClass::byteAlignment * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;               \
      } typedef CPP_TYPE BOOST_PP_CAT(TypeOf_, NAME);                                                                \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::eigen;   \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      CPP_TYPE::Scalar const* BOOST_PP_CAT(addressOf_, NAME)() const {                                               \
        return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                      \
      }                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                         \
      CPP_TYPE::Scalar* BOOST_PP_CAT(addressOf_, NAME)() {                                                           \
        return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                      \
      }                                                                                                              \
)
// clang-format on
#define _DEFINE_METADATA_MEMBERS(R, DATA, TYPE_NAME) _DEFINE_METADATA_MEMBERS_IMPL TYPE_NAME

/**
 * Member assignment for trivial constructor
 */
#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_IMPL(VALUE_TYPE, CPP_TYPE, NAME) \
  _SWITCH_ON_TYPE(VALUE_TYPE,                       /* Scalar */              \
                  (BOOST_PP_CAT(NAME, _)(nullptr)), /* Column */              \
                  (BOOST_PP_CAT(NAME, _)(nullptr)), /* Eigen column */        \
                  (BOOST_PP_CAT(NAME, _)(nullptr))(BOOST_PP_CAT(NAME, Stride_)(0)))

#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_IMPL TYPE_NAME)
/**
 * Computation of the column or scalar pointer location in the memory layout (at SoA construction time)
 */
// clang-format off
#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                 \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                                                                            \
                  BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(curMem);                                        \
                  curMem += (((sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                           \
                  , /* Column */                                                                                      \
                  BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(curMem);                                        \
                  curMem += (((nElements_ * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;              \
                  , /* Eigen column */                                                                                \
                  BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(curMem);                                \
                  curMem += (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment *     \
                            CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                \
                  BOOST_PP_CAT(NAME, Stride_) = (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * \
                                                byteAlignment / sizeof(CPP_TYPE::Scalar);)                            \
  if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)                                               \
    if (reinterpret_cast<intptr_t>(BOOST_PP_CAT(NAME, _)) % byteAlignment)                                            \
      throw std::out_of_range("In layout constructor: misaligned column: " #NAME);
// clang-format on

#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME) _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

/**
 * Computation of the column or scalar size for SoA size computation
 */
// clang-format off
#define _ACCUMULATE_SOA_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                              \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                                                                    \
                  ret += (((sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                      \
                  , /* Column */                                                                              \
                  ret += (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;          \
                  , /* Eigen column */                                                                        \
                  ret += (((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment * \
                         CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;)
// clang-format on

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME) _ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME

/**
 * Direct access to column pointer and indexed access
 */
// clang-format off
#define _DECLARE_SOA_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                     \
  _SWITCH_ON_TYPE(                                                                                                 \
      VALUE_TYPE,                                                                              /* Scalar */        \
      SOA_HOST_DEVICE_INLINE CPP_TYPE& NAME() { return *BOOST_PP_CAT(NAME, _); }, /* Column */                     \
      SOA_HOST_DEVICE_INLINE CPP_TYPE* NAME() {                                                                    \
        return BOOST_PP_CAT(NAME, _);                                                                              \
      } SOA_HOST_DEVICE_INLINE CPP_TYPE& NAME(size_t index) { return BOOST_PP_CAT(NAME, _)[index]; },              \
      /* Eigen column */ /* Unsupported for the moment TODO */                                                     \
      BOOST_PP_EMPTY())
// clang-format on

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

/**
 * Direct access to column pointer (const) and indexed access.
 */
// clang-format off
#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                               \
  _SWITCH_ON_TYPE(                                                                                                 \
      VALUE_TYPE,                                                                                     /* Scalar */ \
      SOA_HOST_DEVICE_INLINE CPP_TYPE NAME() const { return *(BOOST_PP_CAT(NAME, _)); },              /* Column */ \
      SOA_HOST_DEVICE_INLINE CPP_TYPE const* NAME()                                                                \
          const { return BOOST_PP_CAT(NAME, _); } SOA_HOST_DEVICE_INLINE CPP_TYPE NAME(size_t index)               \
              const { return *(BOOST_PP_CAT(NAME, _) + index); }, /* Eigen column */                               \
      SOA_HOST_DEVICE_INLINE CPP_TYPE::Scalar const* NAME()                                                        \
          const { return BOOST_PP_CAT(NAME, _); } SOA_HOST_DEVICE_INLINE size_t BOOST_PP_CAT(                      \
              NAME, Stride)() { return BOOST_PP_CAT(NAME, Stride_); })
// clang-format on

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

/**
 * SoA class member declaration (column pointers).
 */
// clang-format off
#define _DECLARE_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)     \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                            \
                  CPP_TYPE* BOOST_PP_CAT(NAME, _) = nullptr;          \
                  , /* Column */                                      \
                  CPP_TYPE * BOOST_PP_CAT(NAME, _) = nullptr;         \
                  , /* Eigen column */                                \
                  CPP_TYPE::Scalar * BOOST_PP_CAT(NAME, _) = nullptr; \
                  size_t BOOST_PP_CAT(NAME, Stride_) = 0;)
// clang-format on

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

#ifdef DEBUG
#define _DO_RANGECHECK true
#else
#define _DO_RANGECHECK false
#endif

/*
 * A macro defining a SoA layout (collection of scalars and columns of equal lengths)
 */
// clang-format off
#define GENERATE_SOA_LAYOUT(CLASS, ...)                                                                                                   \
  template <size_t ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                                                                      \
            cms::soa::AlignmentEnforcement ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed>                               \
  struct CLASS: public cms::soa::BaseLayout {                                                                                             \
    /* these could be moved to an external type trait to free up the symbol names */                                                      \
    using self_type = CLASS;                                                                                                              \
    typedef cms::soa::AlignmentEnforcement AlignmentEnforcement;                                                                          \
                                                                                                                                          \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                               \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                           \
   * up to compute capability 8.X.                                                                                                        \
   */                                                                                                                                     \
    constexpr static size_t defaultAlignment = 128;                                                                                       \
    constexpr static size_t byteAlignment = ALIGNMENT;                                                                                    \
    constexpr static AlignmentEnforcement alignmentEnforcement = ALIGNMENT_ENFORCEMENT;                                                   \
    constexpr static size_t conditionalAlignment =                                                                                        \
        alignmentEnforcement == AlignmentEnforcement::Enforced ? byteAlignment : 0;                                                       \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                                             \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                                               \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment>;                                                    \
                                                                                                                                          \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                                               \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment>;                                          \
                                                                                                                                          \
    /* dump the SoA internal structure */                                                                                                 \
    SOA_HOST_ONLY                                                                                                                         \
    void toStream(std::ostream & os) const {                                                                                              \
      os << #CLASS "(" << nElements_ << " elements, byte alignement= " << byteAlignment << ", @"<< mem_ <<"): " << std::endl;             \
      os << "  sizeof(" #CLASS "): " << sizeof(CLASS) << std::endl;                                                                       \
      size_t offset = 0;                                                                                                                  \
      _ITERATE_ON_ALL(_DECLARE_SOA_STREAM_INFO, ~, __VA_ARGS__)                                                                           \
      os << "Final offset = " << offset << " computeDataSize(...): " << computeDataSize(nElements_)                                       \
              << std::endl;                                                                                                               \
      os << std::endl;                                                                                                                    \
    }                                                                                                                                     \
                                                                                                                                          \
    /* Helper function used by caller to externally allocate the storage */                                                               \
    static size_t computeDataSize(size_t nElements) {                                                                                     \
      size_t ret = 0;                                                                                                                     \
      _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                                            \
      return ret;                                                                                                                         \
    }                                                                                                                                     \
                                                                                                                                          \
    /**                                                                                                                                   \
   * Helper/friend class allowing SoA introspection.                                                                                      \
   */                                                                                                                                     \
    struct SoAMetadata {                                                                                                                  \
      friend CLASS;                                                                                                                       \
      SOA_HOST_DEVICE_INLINE size_t size() const { return parent_.nElements_; }                                                           \
      SOA_HOST_DEVICE_INLINE size_t byteSize() const { return parent_.byteSize_; }                                                        \
      SOA_HOST_DEVICE_INLINE size_t byteAlignment() const { return CLASS::byteAlignment; }                                                \
      SOA_HOST_DEVICE_INLINE std::byte* data() { return parent_.mem_; }                                                                   \
      SOA_HOST_DEVICE_INLINE const std::byte* data() const { return parent_.mem_; }                                                       \
      SOA_HOST_DEVICE_INLINE std::byte* nextByte() const { return parent_.mem_ + parent_.byteSize_; }                                     \
      SOA_HOST_DEVICE_INLINE CLASS cloneToNewAddress(std::byte* addr) const {                                                             \
        return CLASS(addr, parent_.nElements_);                                                                                           \
      }                                                                                                                                   \
      _ITERATE_ON_ALL(_DEFINE_METADATA_MEMBERS, ~, __VA_ARGS__)                                                                           \
                                                                                                                                          \
      SoAMetadata& operator=(const SoAMetadata&) = delete;                                                                                \
      SoAMetadata(const SoAMetadata&) = delete;                                                                                           \
                                                                                                                                          \
    private:                                                                                                                              \
      SOA_HOST_DEVICE_INLINE SoAMetadata(const CLASS& parent) : parent_(parent) {}                                                        \
      const CLASS& parent_;                                                                                                               \
      typedef CLASS ParentClass;                                                                                                          \
    };                                                                                                                                    \
    friend SoAMetadata;                                                                                                                   \
    SOA_HOST_DEVICE_INLINE const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }                                           \
    SOA_HOST_DEVICE_INLINE SoAMetadata soaMetadata() { return SoAMetadata(*this); }                                                       \
                                                                                                                                          \
    /* Trivial constuctor */                                                                                                              \
    CLASS()                                                                                                                               \
        : mem_(nullptr),                                                                                                                  \
          nElements_(0),                                                                                                                  \
          byteSize_(0),                                                                                                                   \
          _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION, ~, __VA_ARGS__) {}                                                  \
                                                                                                                                          \
    /* Constructor relying on user provided storage */                                                                                    \
    SOA_HOST_ONLY CLASS(std::byte* mem, size_t nElements) : mem_(mem), nElements_(nElements), byteSize_(0) {                              \
      if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)                                                               \
        if (reinterpret_cast<intptr_t>(mem) % byteAlignment)                                                                              \
          throw std::out_of_range("In " #CLASS "::" #CLASS ": misaligned buffer");                                                        \
      auto curMem = mem_;                                                                                                                 \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                       \
      /* Sanity check: we should have reached the computed size, only on host code */                                                     \
      byteSize_ = computeDataSize(nElements_);                                                                                            \
      if (mem_ + byteSize_ != curMem)                                                                                                     \
        throw std::out_of_range("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                                    \
    }                                                                                                                                     \
                                                                                                                                          \
    /* Constructor relying on user provided storage */                                                                                    \
    SOA_DEVICE_ONLY CLASS(bool devConstructor, std::byte* mem, size_t nElements) : mem_(mem), nElements_(nElements) {                     \
      auto curMem = mem_;                                                                                                                 \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                       \
    }                                                                                                                                     \
                                                                                                                                          \
    /* dump the SoA internal structure */                                                                                                 \
    template <typename T>                                                                                                                 \
    SOA_HOST_ONLY friend void dump();                                                                                                     \
                                                                                                                                          \
  private:                                                                                                                                \
    /* Range checker conditional to the macro _DO_RANGECHECK */                                                                           \
    SOA_HOST_DEVICE_INLINE                                                                                                                \
    void rangeCheck(size_t index) const {                                                                                                 \
      if constexpr (_DO_RANGECHECK) {                                                                                                     \
        if (index >= nElements_) {                                                                                                        \
          printf("In " #CLASS "::rangeCheck(): index out of range: %zu with nElements: %zu\n", index, nElements_);                        \
          assert(false);                                                                                                                  \
        }                                                                                                                                 \
      }                                                                                                                                   \
    }                                                                                                                                     \
                                                                                                                                          \
    /* data members */                                                                                                                    \
    std::byte* mem_;                                                                                                                      \
    size_t nElements_;                                                                                                                    \
    size_t byteSize_;                                                                                                                     \
    _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                                             \
  };
// clang-format on

#endif  // ndef DataStructures_SoALayout_h
