/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#ifndef DataStructures_SoAStore_h
#define DataStructures_SoAStore_h

#include "SoACommon.h"

#include <iostream>
#include <cassert>

/* dump SoA fields information; these should expand to, for columns:
 * Example:
 * generate_SoA_store(SoA,
 *   // predefined static scalars
 *   // size_t size;
 *   // size_t alignment;
 *
 *   // columns: one value per element
 *   SoA_FundamentalTypeColumn(double, x),
 *   SoA_FundamentalTypeColumn(double, y),
 *   SoA_FundamentalTypeColumn(double, z),
 *   SoA_eigenColumn(Eigen::Vector3d, a),
 *   SoA_eigenColumn(Eigen::Vector3d, b),
 *   SoA_eigenColumn(Eigen::Vector3d, r),
 *   SoA_column(uint16_t, colour),
 *   SoA_column(int32_t, value),
 *   SoA_column(double *, py),
 *   SoA_FundamentalTypeColumn(uint32_t, count),
 *   SoA_FundamentalTypeColumn(uint32_t, anotherCount),
 *
 *   // scalars: one value for the whole structure
 *   SoA_scalar(const char *, description),
 *   SoA_scalar(uint32_t, someNumber)
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

#define _DECLARE_SOA_DUMP_INFO_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                       \
  _SWITCH_ON_TYPE(                                                                                                    \
      VALUE_TYPE, /* Dump scalar */                                                                                   \
      std::cout << " Scalar " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset << " has size " << sizeof(CPP_TYPE) << " and padding "                       \
                               << ((sizeof(CPP_TYPE) - 1) / byteAlignment + 1) * byteAlignment - sizeof(CPP_TYPE)     \
                               << std::endl;                                                                          \
          offset += ((sizeof(CPP_TYPE) - 1) / byteAlignment + 1) * byteAlignment;                                     \
          , /* Dump column */                                                                                         \
          std::cout                                                                                                   \
                << " Column " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset << " has size " << sizeof(CPP_TYPE) * nElements                 \
                                            << " and padding "                                                        \
                                            << (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) *           \
                                                       byteAlignment -                                                \
                                                   (sizeof(CPP_TYPE) * nElements)                                     \
                                            << std::endl;                                                             \
              offset += (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                   \
              , /* Dump Eigen column */                                                                               \
              std::cout                                                                                               \
                << " Eigen value " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset << " has dimension (" << CPP_TYPE::RowsAtCompileTime << " x "   \
                                            << CPP_TYPE::ColsAtCompileTime                                            \
                                            << ")"                                                                    \
                                            << " and per column size "                                                \
                                            << sizeof(CPP_TYPE::Scalar) * nElements                                   \
                                            << " and padding "                                                        \
                                            << (((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) *   \
                                                       byteAlignment -                                                \
                                                   (sizeof(CPP_TYPE::Scalar) * nElements)                             \
                                            << std::endl;                                                             \
                       offset += (((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment * \
                                 CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;)

#define _DECLARE_SOA_DUMP_INFO(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_DUMP_INFO_IMPL TYPE_NAME)

/**
 * SoAMetadata member computing column pitch
 */
#define _DEFINE_METADATA_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                     \
  _SWITCH_ON_TYPE(                                                                                                    \
      VALUE_TYPE, /* Scalar */                                                                                        \
      size_t BOOST_PP_CAT(NAME, Pitch()) const {                                                                      \
        return (((sizeof(CPP_TYPE) - 1) / parent_.byteAlignment_) + 1) * parent_.byteAlignment_;                      \
      } typedef CPP_TYPE BOOST_PP_CAT(TypeOf_, NAME);                                                                 \
      static const SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = SoAColumnType::scalar;                           \
      CPP_TYPE * BOOST_PP_CAT(addressOf_, NAME)() const { return parent_.BOOST_PP_CAT(NAME, _); }, /* Column */       \
      size_t BOOST_PP_CAT(NAME, Pitch()) const {                                                                      \
        return (((parent_.nElements_ * sizeof(CPP_TYPE) - 1) / parent_.byteAlignment_) + 1) * parent_.byteAlignment_; \
      } typedef CPP_TYPE BOOST_PP_CAT(TypeOf_, NAME);                                                                 \
      static const SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = SoAColumnType::column;                           \
      CPP_TYPE * BOOST_PP_CAT(addressOf_, NAME)() const { return parent_.BOOST_PP_CAT(NAME, _); }, /* Eigen column */ \
      size_t BOOST_PP_CAT(NAME, Pitch()) const {                                                                      \
        return (((parent_.nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / parent_.byteAlignment_) + 1) *                 \
               parent_.byteAlignment_ * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                    \
      } typedef CPP_TYPE BOOST_PP_CAT(TypeOf_, NAME);                                                                 \
      static const SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = SoAColumnType::eigen;                            \
      CPP_TYPE::Scalar * BOOST_PP_CAT(addressOf_, NAME)() const { return parent_.BOOST_PP_CAT(NAME, _); })

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
#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                                  \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                                                                             \
                  BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(curMem);                                         \
                  curMem += (((sizeof(CPP_TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                          \
                  , /* Column */                                                                                       \
                  BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(curMem);                                         \
                  curMem += (((nElements_ * sizeof(CPP_TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;             \
                  , /* Eigen column */                                                                                 \
                  BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(curMem);                                 \
                  curMem += (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment_) + 1) * byteAlignment_ *    \
                            CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                 \
                  BOOST_PP_CAT(NAME, Stride_) = (((nElements_ * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment_) + 1) * \
                                                byteAlignment_ / sizeof(CPP_TYPE::Scalar);)

#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME) _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

/**
 * Computation of the column or scalar size for SoA size computation
 */
#define _ACCUMULATE_SOA_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                              \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                                                                    \
                  ret += (((sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                      \
                  , /* Column */                                                                              \
                  ret += (((nElements * sizeof(CPP_TYPE) - 1) / byteAlignment) + 1) * byteAlignment;          \
                  , /* Eigen column */                                                                        \
                  ret += (((nElements * sizeof(CPP_TYPE::Scalar) - 1) / byteAlignment) + 1) * byteAlignment * \
                         CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;)

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME) _ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME

/**
 * Value accessor of the const_element subclass.
 */
#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                      \
  SOA_HOST_DEVICE_INLINE                                                                                          \
  _SWITCH_ON_TYPE(                                                                                                \
      VALUE_TYPE,                                     /* Scalar */                                                \
      CPP_TYPE const& NAME() { return soa_.NAME(); }, /* Column */                                                \
      CPP_TYPE const& NAME() { return *(soa_.NAME() + index_); },                                                 \
      /* Eigen column */ /* Ugly hack with a helper template to avoid having commas inside the macro parameter */ \
      EigenConstMapMaker<CPP_TYPE>::Type const NAME() {                                                           \
        return EigenConstMapMaker<CPP_TYPE>::withData(soa_.NAME() + index_)                                       \
            .withStride(soa_.BOOST_PP_CAT(NAME, Stride)());                                                       \
      })

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME) _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME

/**
 * Generator of parameters for (non-const) element subclass (expanded comma separated).
 */
#define _DECLARE_ELEMENT_VALUE_ARG_IMPL(VALUE_TYPE, CPP_TYPE, NAME) \
  _SWITCH_ON_TYPE(VALUE_TYPE,        /* Scalar */                   \
                  BOOST_PP_EMPTY(),  /* Column */                   \
                  (CPP_TYPE * NAME), /* Eigen column */             \
                  (CPP_TYPE::Scalar * NAME)(size_t BOOST_PP_CAT(NAME, Stride)))

#define _DECLARE_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME) _DECLARE_ELEMENT_VALUE_ARG_IMPL TYPE_NAME

/**
 * Generator of member initialization for constructor of element subclass
 */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL(VALUE_TYPE, CPP_TYPE, NAME, DATA) \
  _SWITCH_ON_TYPE(VALUE_TYPE,         /* Scalar */                                          \
                  BOOST_PP_EMPTY(),   /* Column */                                          \
                  (NAME(DATA, NAME)), /* Eigen column */                                    \
                  (NAME(DATA, NAME, BOOST_PP_CAT(NAME, Stride))))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))

/**
 * Generator of member initialization for constructor of const element subclass
 */
#define _DECLARE_CONST_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL(VALUE_TYPE, CPP_TYPE, NAME, DATA) \
  _SWITCH_ON_TYPE(VALUE_TYPE,                          /* Scalar */                               \
                  BOOST_PP_EMPTY(),                    /* Column */                               \
                  (BOOST_PP_CAT(NAME, _)(DATA, NAME)), /* Eigen column */                         \
                  (BOOST_PP_CAT(NAME, _)(DATA, NAME, BOOST_PP_CAT(NAME, Stride))))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_CONST_ELEMENT_VALUE_MEMBER_INITIALISATION(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_CONST_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))
/**
 * Generator of the member-by-member copy operator of the element subclass.
 */
#define _DECLARE_ELEMENT_VALUE_COPY_IMPL(VALUE_TYPE, CPP_TYPE, NAME) \
  _SWITCH_ON_TYPE(VALUE_TYPE,       /* Scalar */                     \
                  BOOST_PP_EMPTY(), /* Column */                     \
                  NAME() = other.NAME();                             \
                  , /* Eigen column */                               \
                  static_cast<CPP_TYPE>(NAME) = static_cast<std::add_const<CPP_TYPE>::type&>(other.NAME);)

#define _DECLARE_ELEMENT_VALUE_COPY(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_COPY_IMPL TYPE_NAME)

/**
 * Declaration of the private members of the const element subclass
 */
#define _DECLARE_CONST_ELEMENT_VALUE_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME) \
  _SWITCH_ON_TYPE(VALUE_TYPE,       /* Scalar */                             \
                  BOOST_PP_EMPTY(), /* Column */                             \
                  const SoAValue<CPP_TYPE> BOOST_PP_CAT(NAME, _);            \
                  , /* Eigen column */                                       \
                  const SoAEigenValue<CPP_TYPE> BOOST_PP_CAT(NAME, _);)

#define _DECLARE_CONST_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME) _DECLARE_CONST_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME

/**
 * Declaration of the members accessors of the const element subclass
 */
#define _DECLARE_CONST_ELEMENT_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                   \
  _SWITCH_ON_TYPE(                                                                                         \
      VALUE_TYPE,                                                                       /* Scalar */       \
      BOOST_PP_EMPTY(),                                                                 /* Column */       \
      SOA_HOST_DEVICE_INLINE CPP_TYPE NAME() const { return BOOST_PP_CAT(NAME, _)(); }, /* Eigen column */ \
      SOA_HOST_DEVICE_INLINE const SoAEigenValue<CPP_TYPE> NAME() const { return BOOST_PP_CAT(NAME, _); })

#define _DECLARE_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME) _DECLARE_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME

/**
 * Declaration of the members of the element subclass
 */
#define _DECLARE_ELEMENT_VALUE_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME) \
  _SWITCH_ON_TYPE(VALUE_TYPE,       /* Scalar */                       \
                  BOOST_PP_EMPTY(), /* Column */                       \
                  SoAValue<CPP_TYPE> NAME;                             \
                  , /* Eigen column */                                 \
                  SoAEigenValue<CPP_TYPE> NAME;)

#define _DECLARE_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME) _DECLARE_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME

/**
 * Parameters passed to element subclass constructor in operator[]
 */
#define _DECLARE_ELEMENT_CONSTR_CALL_IMPL(VALUE_TYPE, CPP_TYPE, NAME) \
  _SWITCH_ON_TYPE(VALUE_TYPE,              /* Scalar */               \
                  BOOST_PP_EMPTY(),        /* Column */               \
                  (BOOST_PP_CAT(NAME, _)), /* Eigen column */         \
                  (BOOST_PP_CAT(NAME, _))(BOOST_PP_CAT(NAME, Stride_)))

#define _DECLARE_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME)

/**
 * Direct access to column pointer and indexed access
 */
#define _DECLARE_SOA_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                        \
  _SWITCH_ON_TYPE(                                                                                    \
      VALUE_TYPE,                                                                 /* Scalar */        \
      SOA_HOST_DEVICE_INLINE CPP_TYPE& NAME() { return *BOOST_PP_CAT(NAME, _); }, /* Column */        \
      SOA_HOST_DEVICE_INLINE CPP_TYPE* NAME() {                                                       \
        return BOOST_PP_CAT(NAME, _);                                                                 \
      } SOA_HOST_DEVICE_INLINE CPP_TYPE& NAME(size_t index) { return BOOST_PP_CAT(NAME, _)[index]; }, \
      /* Eigen column */ /* Unsupported for the moment TODO */                                        \
      BOOST_PP_EMPTY())

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

/**
 * Direct access to column pointer (const) and indexed access.
 */
#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                  \
  _SWITCH_ON_TYPE(                                                                                    \
      VALUE_TYPE,                                                                        /* Scalar */ \
      SOA_HOST_DEVICE_INLINE CPP_TYPE NAME() const { return *(BOOST_PP_CAT(NAME, _)); }, /* Column */ \
      SOA_HOST_DEVICE_INLINE CPP_TYPE const* NAME()                                                   \
          const { return BOOST_PP_CAT(NAME, _); } SOA_HOST_DEVICE_INLINE CPP_TYPE NAME(size_t index)  \
              const { return *(BOOST_PP_CAT(NAME, _) + index); }, /* Eigen column */                  \
      SOA_HOST_DEVICE_INLINE CPP_TYPE::Scalar const* NAME() const {                                   \
        return BOOST_PP_CAT(NAME, _);                                                                 \
      } SOA_HOST_DEVICE_INLINE size_t BOOST_PP_CAT(NAME, Stride)() { return BOOST_PP_CAT(NAME, Stride_); })

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

/**
 * SoA class member declaration (column pointers).
 */
#define _DECLARE_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME)     \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                            \
                  CPP_TYPE* BOOST_PP_CAT(NAME, _) = nullptr;          \
                  , /* Column */                                      \
                  CPP_TYPE * BOOST_PP_CAT(NAME, _) = nullptr;         \
                  , /* Eigen column */                                \
                  CPP_TYPE::Scalar * BOOST_PP_CAT(NAME, _) = nullptr; \
                  size_t BOOST_PP_CAT(NAME, Stride_) = 0;)

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

#ifdef DEBUG
#define _DO_RANGECHECK true
#else
#define _DO_RANGECHECK false
#endif

/*
 * A macro defining a SoA store (collection of scalars and columns of equal lengths
 */
#define generate_SoA_store(CLASS, ...)                                                                                                  \
  struct CLASS {                                                                                                                        \
    /* these could be moved to an external type trait to free up the symbol names */                                                    \
    using self_type = CLASS;                                                                                                            \
                                                                                                                                        \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                           \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                     \
   * up to compute capability 8.X.                                                                                                  \
   */ \
    constexpr static size_t defaultAlignment = 128;                                                                                     \
                                                                                                                                        \
    /* dump the SoA internal structure */                                                                                               \
    SOA_HOST_ONLY                                                                                                                       \
    static void dump(size_t nElements, size_t byteAlignment = defaultAlignment) {                                                       \
      std::cout << #CLASS "(" << nElements << ", " << byteAlignment << "): " << std::endl;                                              \
      std::cout << "  sizeof(" #CLASS "): " << sizeof(CLASS) << std::endl;                                                              \
      size_t offset = 0;                                                                                                                \
      _ITERATE_ON_ALL(_DECLARE_SOA_DUMP_INFO, ~, __VA_ARGS__)                                                                           \
      std::cout << "Final offset = " << offset                                                                                          \
                << " computeDataSize(...): " << computeDataSize(nElements, byteAlignment) << std::endl;                                 \
      std::cout << std::endl;                                                                                                           \
    }                                                                                                                                   \
    /* Helper function used by caller to externally allocate the storage */                                                             \
    static size_t computeDataSize(size_t nElements, size_t byteAlignment = defaultAlignment) {                                          \
      size_t ret = 0;                                                                                                                   \
      _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                                          \
      return ret;                                                                                                                       \
    }                                                                                                                                   \
                                                                                                                                        \
    /**                                                                                                                               \
   * Helper/friend class allowing SoA introspection.                                                                                \
   */ \
    struct SoAMetadata {                                                                                                                \
      friend CLASS;                                                                                                                     \
      SOA_HOST_DEVICE_INLINE size_t size() const { return parent_.nElements_; }                                                         \
      SOA_HOST_DEVICE_INLINE size_t byteSize() const { return parent_.byteSize_; }                                                      \
      SOA_HOST_DEVICE_INLINE size_t byteAlignment() const { return parent_.byteAlignment_; }                                            \
      SOA_HOST_DEVICE_INLINE std::byte* data() const { return parent_.mem_; }                                                           \
      SOA_HOST_DEVICE_INLINE std::byte* nextByte() const { return parent_.mem_ + parent_.byteSize_; }                                   \
      SOA_HOST_DEVICE_INLINE CLASS cloneToNewAddress(std::byte* addr) {                                                                 \
        return CLASS(addr, parent_.nElements_, parent_.byteAlignment_);                                                                 \
      }                                                                                                                                 \
      _ITERATE_ON_ALL(_DEFINE_METADATA_MEMBERS, ~, __VA_ARGS__)                                                                         \
                                                                                                                                        \
    private:                                                                                                                            \
      SOA_HOST_DEVICE_INLINE SoAMetadata(const CLASS& parent) : parent_(parent) {}                                                      \
      const CLASS& parent_;                                                                                                             \
    };                                                                                                                                  \
    friend SoAMetadata;                                                                                                                 \
    SOA_HOST_DEVICE_INLINE const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }                                         \
                                                                                                                                        \
    /* Trivial constuctor */                                                                                                            \
    CLASS() : _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION, ~, __VA_ARGS__) {}                                            \
                                                                                                                                        \
    /* Constructor relying on user provided storage */                                                                                  \
    SOA_HOST_ONLY CLASS(std::byte* mem, size_t nElements, size_t byteAlignment = defaultAlignment)                                      \
        : mem_(mem), nElements_(nElements), byteAlignment_(byteAlignment) {                                                             \
      auto curMem = mem_;                                                                                                               \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                     \
      /* Sanity check: we should have reached the computed size, only on host code */                                                   \
      byteSize_ = computeDataSize(nElements_, byteAlignment_);                                                                          \
      if (mem_ + byteSize_ != curMem)                                                                                                   \
        throw std::out_of_range("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                                  \
    }                                                                                                                                   \
                                                                                                                                        \
    /* Constructor relying on user provided storage */                                                                                  \
    SOA_DEVICE_ONLY CLASS(bool devConstructor,                                                                                          \
                          std::byte* mem,                                                                                               \
                          size_t nElements,                                                                                             \
                          size_t byteAlignment = defaultAlignment)                                                                      \
        : mem_(mem), nElements_(nElements), byteAlignment_(byteAlignment) {                                                             \
      auto curMem = mem_;                                                                                                               \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                     \
    }                                                                                                                                   \
                                                                                                                                        \
    struct const_element {                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      const_element(size_t index, /* Declare parameters */                                                                              \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_VALUE_ARG, index, __VA_ARGS__))                                              \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_CONST_ELEMENT_VALUE_MEMBER_INITIALISATION, index, __VA_ARGS__) {}                            \
      _ITERATE_ON_ALL(_DECLARE_CONST_ELEMENT_ACCESSOR, ~, __VA_ARGS__)                                                                  \
                                                                                                                                        \
    private:                                                                                                                            \
      _ITERATE_ON_ALL(_DECLARE_CONST_ELEMENT_VALUE_MEMBER, ~, __VA_ARGS__)                                                              \
    };                                                                                                                                  \
                                                                                                                                        \
    struct element {                                                                                                                    \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      element(size_t index, /* Declare parameters */                                                                                    \
              _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_VALUE_ARG, index, __VA_ARGS__))                                                    \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION, index, __VA_ARGS__) {}                                  \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      element& operator=(const element& other) {                                                                                        \
        _ITERATE_ON_ALL(_DECLARE_ELEMENT_VALUE_COPY, ~, __VA_ARGS__)                                                                    \
        return *this;                                                                                                                   \
      }                                                                                                                                 \
      _ITERATE_ON_ALL(_DECLARE_ELEMENT_VALUE_MEMBER, ~, __VA_ARGS__)                                                                    \
    };                                                                                                                                  \
                                                                                                                                        \
    /* AoS-like accessor (non-const) */                                                                                                 \
    SOA_HOST_DEVICE_INLINE                                                                                                              \
    element operator[](size_t index) {                                                                                                  \
      rangeCheck(index);                                                                                                                \
      return element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, __VA_ARGS__));                                       \
    }                                                                                                                                   \
                                                                                                                                        \
    /* AoS-like accessor (const) */                                                                                                     \
    SOA_HOST_DEVICE_INLINE                                                                                                              \
    const const_element operator[](size_t index) const {                                                                                \
      rangeCheck(index);                                                                                                                \
      return const_element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, __VA_ARGS__));                                 \
    }                                                                                                                                   \
                                                                                                                                        \
    /* accessors */                                                                                                                     \
    _ITERATE_ON_ALL(_DECLARE_SOA_ACCESSOR, ~, __VA_ARGS__)                                                                              \
    _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ACCESSOR, ~, __VA_ARGS__)                                                                        \
                                                                                                                                        \
    /* dump the SoA internal structure */                                                                                               \
    template <typename T>                                                                                                               \
    SOA_HOST_ONLY friend void dump();                                                                                                   \
                                                                                                                                        \
  private:                                                                                                                              \
    /* Range checker conditional to the macro _DO_RANGECHECK */                                                                         \
    SOA_HOST_DEVICE_INLINE                                                                                                              \
    void rangeCheck(size_t index) const {                                                                                               \
      if constexpr (_DO_RANGECHECK) {                                                                                                   \
        if (index >= nElements_) {                                                                                                      \
          printf("In " #CLASS "::rangeCheck(): index out of range: %zu with nElements: %zu\n", index, nElements_);                      \
          assert(false);                                                                                                                \
        }                                                                                                                               \
      }                                                                                                                                 \
    }                                                                                                                                   \
                                                                                                                                        \
    /* data members */                                                                                                                  \
    std::byte* mem_;                                                                                                                    \
    size_t nElements_;                                                                                                                  \
    size_t byteSize_;                                                                                                                   \
    size_t byteAlignment_;                                                                                                              \
    _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                                           \
  }

#endif  // ndef DataStructures_SoAStore_h
