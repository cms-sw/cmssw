#ifndef DataFormats_SoATemplate_interface_SoALayout_h
#define DataFormats_SoATemplate_interface_SoALayout_h

/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#include "FWCore/Reflection/interface/reflex.h"

#include "SoACommon.h"

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

namespace cms::soa {

  /* Traits for the different column type scenarios */
  /* Value traits passes the class as is in the case of column type and return
   * an empty class with functions returning non-scalar as accessors. */
  template <class C, SoAColumnType COLUMN_TYPE>
  struct ConstValueTraits : public C {
    using C::C;
  };

  template <class C>
  struct ConstValueTraits<C, SoAColumnType::scalar> {
    // Just take to SoAValue type to generate the right constructor.
    SOA_HOST_DEVICE SOA_INLINE ConstValueTraits(size_type, const typename C::valueType*) {}
    SOA_HOST_DEVICE SOA_INLINE ConstValueTraits(size_type, const typename C::Params&) {}
    SOA_HOST_DEVICE SOA_INLINE ConstValueTraits(size_type, const typename C::ConstParams&) {}
    // Any attempt to do anything with the "scalar" value a const element will fail.
  };

  /* Helper to extract the column type from a ConstValue type
   * and so to avoid commas inside macros. */
  template <typename C>
  struct ColumnTypeOf;

  template <SoAColumnType CT, typename T, byte_size_type ALIGNMENT, bool RESTRICT>
  struct ColumnTypeOf<SoAConstValue<CT, T, ALIGNMENT, RESTRICT>> {
    static constexpr SoAColumnType value = CT;
  };

  template <typename C>
  using ConstValueTraitsFromC = cms::soa::ConstValueTraits<C, ColumnTypeOf<C>::value>;
}  // namespace cms::soa
// namespace cms::soa

// -------- MACROS FOR GENERATING SOA LAYOUT --------

#define _COUNT_SOA_METHODS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS, DATA) \
  BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_METHOD), DATA++;, BOOST_PP_EMPTY())

#define _COUNT_SOA_METHODS(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_COUNT_SOA_METHODS_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))

#define _COUNT_SOA_CONST_METHODS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS, DATA) \
  BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_CONST_METHOD), DATA++;, BOOST_PP_EMPTY())

#define _COUNT_SOA_CONST_METHODS(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_COUNT_SOA_CONST_METHODS_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))

// clang-format off
#define _DECLARE_SOA_STREAM_INFO_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                                \
  cms::soa::detail::printColumn(_soa_impl_os, ConstView::BOOST_PP_CAT(NAME, Parameters_), BOOST_PP_STRINGIZE(NAME), _soa_impl_offset, elements_, alignment);
// clang-format on

#define _DECLARE_SOA_STREAM_INFO(R, DATA, TYPE_NAME)                                        \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_SOA_STREAM_INFO_IMPL TYPE_NAME))

/**
 * Metadata member computing column pitch
 */
// clang-format off
#define _DEFINE_METADATA_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                                \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::scalar;    \
      using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                    \
        cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::scalar>::DataType<CPP_TYPE>;                       \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                                \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (parent_.BOOST_PP_CAT(NAME, _));                                 \
      },                                                                                                               \
      /* Column */                                                                                                     \
      using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                    \
        cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<CPP_TYPE>;                       \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                                \
        return  BOOST_PP_CAT(ParametersTypeOf_, NAME) (parent_.BOOST_PP_CAT(NAME, _));                                 \
      }                                                                                                                \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::column;,   \
      /* Eigen column */                                                                                               \
      using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                    \
          cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::eigen>::DataType<CPP_TYPE>;                      \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(parametersOf_, NAME)() const {                                \
        return BOOST_PP_CAT(ParametersTypeOf_, NAME) (                                                                 \
          parent_.BOOST_PP_CAT(NAME, _),                                                                               \
          parent_.BOOST_PP_CAT(NAME, Stride_));                                                                        \
      }                                                                                                                \
      constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) = cms::soa::SoAColumnType::eigen;     \
  )																													   \
  SOA_HOST_DEVICE SOA_INLINE                                                                                       	   \
  auto* BOOST_PP_CAT(addressOf_, NAME)() {                                                                     		   \
	return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           	   \
  }                                                                                                               	   \
  SOA_HOST_DEVICE SOA_INLINE                                                                                       	   \
  const auto* BOOST_PP_CAT(addressOf_, NAME)() const {                                                         		   \
	return parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                           	   \
  } 																												   \
  SOA_HOST_DEVICE SOA_INLINE byte_size_type BOOST_PP_CAT(NAME, Pitch()) const {                                        \
	return cms::soa::detail::computePitch(parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)(),					   \
						ParentClass::alignment, parent_.elements_);													   \
  }
// clang-format on

#define _DEFINE_METADATA_MEMBERS(R, DATA, TYPE_NAME)                                        \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DEFINE_METADATA_MEMBERS_IMPL TYPE_NAME))

/**
 * Declare the spans of the const descriptor data member 
 */
// clang-format off
#define _DECLARE_CONST_DESCRIPTOR_SPANS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                        \
  (cms::soa::detail::ConstSpanType<typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)>)
// clang-format on

#define _DECLARE_CONST_DESCRIPTOR_SPANS(R, DATA, TYPE_NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONST_DESCRIPTOR_SPANS_IMPL TYPE_NAME))

/**
 * Declare const descriptor parameter types 
 */
// clang-format off
#define _DECLARE_CONST_DESCRIPTOR_PARAMETER_TYPES_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)            \
  (_SWITCH_ON_TYPE(VALUE_TYPE,                                                                      \
      cms::soa::SoAConstParameters_ColumnType<cms::soa::SoAColumnType::scalar>::DataType<CPP_TYPE>, \
      cms::soa::SoAConstParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<CPP_TYPE>, \
      cms::soa::SoAConstParameters_ColumnType<cms::soa::SoAColumnType::eigen>::DataType<CPP_TYPE>)  \
  )
// clang-format on

/**
 * Declare descriptor parameter types 
 */
// clang-format off
#define _DECLARE_DESCRIPTOR_PARAMETER_TYPES_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)             \
  (_SWITCH_ON_TYPE(VALUE_TYPE,                                                                 \
      cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::scalar>::DataType<CPP_TYPE>, \
      cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::column>::DataType<CPP_TYPE>, \
      cms::soa::SoAParameters_ColumnType<cms::soa::SoAColumnType::eigen>::DataType<CPP_TYPE>)  \
  )
// clang-format on

#define _DECLARE_CONST_DESCRIPTOR_PARAMETER_TYPES(R, DATA, TYPE_NAME)                       \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONST_DESCRIPTOR_PARAMETER_TYPES_IMPL TYPE_NAME))

#define _DECLARE_DESCRIPTOR_PARAMETER_TYPES(R, DATA, TYPE_NAME)                             \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_DESCRIPTOR_PARAMETER_TYPES_IMPL TYPE_NAME))

// clang-format off
#define _ASSIGN_PARAMETER_TO_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                 \
  (view.metadata().BOOST_PP_CAT(parametersOf_, NAME)())
// clang-format on

#define _ASSIGN_PARAMETER_TO_COLUMNS(R, DATA, TYPE_NAME)                                    \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_ASSIGN_PARAMETER_TO_COLUMNS_IMPL TYPE_NAME))
/**
 * Declare the spans of the descriptor data member 
 */
// clang-format off
#define _DECLARE_DESCRIPTOR_SPANS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                              \
  (cms::soa::detail::SpanType<typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)>)
// clang-format on

#define _DECLARE_DESCRIPTOR_SPANS(R, DATA, TYPE_NAME)                                       \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_DESCRIPTOR_SPANS_IMPL TYPE_NAME))

/**
 * Build the spans of the (const) descriptor from a (const) view 
 */
// clang-format off
#define _ASSIGN_SPAN_TO_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                                		   \
    (cms::soa::detail::getSpanToColumn(view.metadata().BOOST_PP_CAT(parametersOf_, NAME)(), view.metadata().size(), alignment))
// clang-format on

#define _ASSIGN_SPAN_TO_COLUMNS(R, DATA, TYPE_NAME)                                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_ASSIGN_SPAN_TO_COLUMNS_IMPL TYPE_NAME))

#define _DECLARE_COLUMN_TYPE_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)  \
  BOOST_PP_IF(BOOST_PP_GREATER(VALUE_TYPE, _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                      \
              (cms::soa::SoAColumnType::BOOST_PP_IF(                 \
                  BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_SCALAR),    \
                  scalar,                                            \
                  BOOST_PP_IF(BOOST_PP_EQUAL(VALUE_TYPE, _VALUE_TYPE_COLUMN), column, eigen))))

#define _DECLARE_COLUMN_TYPE(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_DECLARE_COLUMN_TYPE_IMPL TYPE_NAME)

// clang-format off
#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                    \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      (BOOST_PP_CAT(NAME, _)(nullptr)),                                                                                \
      /* Column */                                                                                                     \
      (BOOST_PP_CAT(NAME, _)(nullptr)),                                                                                \
      /* Eigen column */                                                                                               \
      (BOOST_PP_CAT(NAME, ElementsWithPadding_)(0))                                                                    \
      (BOOST_PP_CAT(NAME, _)(nullptr))                                                                                 \
      (BOOST_PP_CAT(NAME, Stride_)(0))                                                                                 \
)
// clang-format on

#define _DECLARE_MEMBER_TRIVIAL_CONSTRUCTION(R, DATA, TYPE_NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION_IMPL TYPE_NAME))

// clang-format off
#define _DECLARE_MEMBER_COPY_CONSTRUCTION_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                       \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      (BOOST_PP_CAT(NAME, _){_soa_impl_other.BOOST_PP_CAT(NAME, _)}),                                                  \
      /* Column */                                                                                                     \
      (BOOST_PP_CAT(NAME, _){_soa_impl_other.BOOST_PP_CAT(NAME, _)}),                                                  \
      /* Eigen column */                                                                                               \
      (BOOST_PP_CAT(NAME, ElementsWithPadding_){_soa_impl_other.BOOST_PP_CAT(NAME, ElementsWithPadding_)})             \
      (BOOST_PP_CAT(NAME, _){_soa_impl_other.BOOST_PP_CAT(NAME, _)})                                                   \
      (BOOST_PP_CAT(NAME, Stride_){_soa_impl_other.BOOST_PP_CAT(NAME, Stride_)})                                       \
  )
// clang-format on

#define _DECLARE_MEMBER_COPY_CONSTRUCTION(R, DATA, TYPE_NAME)                               \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_MEMBER_COPY_CONSTRUCTION_IMPL TYPE_NAME))

// clang-format off
#define _DECLARE_MEMBER_ASSIGNMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                              \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);,                                                  \
      /* Column */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);,                                                  \
      /* Eigen column */                                                                                               \
      BOOST_PP_CAT(NAME, ElementsWithPadding_) = _soa_impl_other.BOOST_PP_CAT(NAME, ElementsWithPadding_);             \
      BOOST_PP_CAT(NAME, _) = _soa_impl_other.BOOST_PP_CAT(NAME, _);                                                   \
      BOOST_PP_CAT(NAME, Stride_) = _soa_impl_other.BOOST_PP_CAT(NAME, Stride_);                                       \
  )
// clang-format on

#define _DECLARE_MEMBER_ASSIGNMENT(R, DATA, TYPE_NAME)                                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_MEMBER_ASSIGNMENT_IMPL TYPE_NAME))

/**
 * Declare the const_cast version of the columns
 * This is used to convert a ConstView into a View
 */
#define _DECLARE_CONST_CAST_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (cms::soa::const_cast_SoAParametersImpl(view.metadata().BOOST_PP_CAT(parametersOf_, NAME)()))

#define _DECLARE_CONST_CAST_COLUMNS(R, DATA, TYPE_NAME)                                     \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONST_CAST_COLUMNS_IMPL TYPE_NAME))

/**
 * Declare the value_element data members
 */
// clang-format off
#define _DEFINE_VALUE_ELEMENT_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                           \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      CPP_TYPE NAME;                                                                                                   \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      CPP_TYPE NAME;                                                                                                   \
  )
// clang-format on

#define _DEFINE_VALUE_ELEMENT_MEMBERS(R, DATA, TYPE_NAME)                                   \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DEFINE_VALUE_ELEMENT_MEMBERS_IMPL TYPE_NAME))

/**
 * List of data members in the value_element constructor arguments
 */
// clang-format off
#define _VALUE_ELEMENT_CTOR_ARGS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                                \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      (CPP_TYPE NAME)                                                                                                  \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      (CPP_TYPE NAME)                                                                                                  \
  )
// clang-format on

#define _VALUE_ELEMENT_CTOR_ARGS(R, DATA, TYPE_NAME)                                        \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_VALUE_ELEMENT_CTOR_ARGS_IMPL TYPE_NAME))

/**
 * List-initalise the value_element data members
 */
// clang-format off
#define _VALUE_ELEMENT_INITIALIZERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                             \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      (NAME{NAME})                                                                                                     \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      (NAME{NAME})                                                                                                     \
  )
// clang-format on

#define _VALUE_ELEMENT_INITIALIZERS(R, DATA, TYPE_NAME)                                     \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_VALUE_ELEMENT_INITIALIZERS_IMPL TYPE_NAME))

/**
 * Freeing of the ROOT-allocated column or scalar buffer
 */
#define _ROOT_FREE_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  delete[] BOOST_PP_CAT(NAME, _);                                              \
  BOOST_PP_CAT(NAME, _) = nullptr;

#define _ROOT_FREE_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_ROOT_FREE_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME))

/**
 * Computation of the column or scalar pointer location in the memory layout (at SoA construction time)
 */
// clang-format off
#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                            \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(sizeof(CPP_TYPE), alignment);                                            \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE*>(_soa_impl_curMem);                                           \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE), alignment);                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      BOOST_PP_CAT(NAME, Stride_) = cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)               \
        / sizeof(CPP_TYPE::Scalar);                                                                                    \
      BOOST_PP_CAT(NAME, ElementsWithPadding_) = BOOST_PP_CAT(NAME, Stride_)                                           \
        *  CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                  \
      BOOST_PP_CAT(NAME, _) = reinterpret_cast<CPP_TYPE::Scalar*>(_soa_impl_curMem);                                   \
      _soa_impl_curMem += cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                         \
        * CPP_TYPE::RowsAtCompileTime * CPP_TYPE::ColsAtCompileTime;                                                   \
  )                                                                                                                    \
  if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                                \
    if (reinterpret_cast<intptr_t>(BOOST_PP_CAT(NAME, _)) % alignment)                                                 \
      throw std::runtime_error("In layout constructor: misaligned column: " #NAME);
// clang-format on

#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME)                                    \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME))

/**
 * Computation of the column or scalar size for SoA size computation
 */
// clang-format off
#define _ACCUMULATE_SOA_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                                 \
  _soa_impl_ret += cms::soa::detail::AccumulateColumnByteSizes<BOOST_PP_CAT(typename Metadata::ParametersTypeOf_, NAME)>{}(elements, alignment);
// clang-format on

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME)                                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME))

/**
 * Direct access to column pointer and indexed access
 */
// clang-format off
#define _DECLARE_SOA_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                                   \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE& NAME() { return *BOOST_PP_CAT(NAME, _); }                                   \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                    \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE& NAME(size_type _soa_impl_index) {                                           \
        return BOOST_PP_CAT(NAME, _)[_soa_impl_index];                                                                 \
      }                                                                                                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      /* TODO: implement*/                                                                                             \
      BOOST_PP_EMPTY()                                                                                                 \
  )
// clang-format on

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                           \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME))

/**
 * Direct access to column pointer (const) and indexed access.
 */
// clang-format off
#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                             \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE NAME() const { return *(BOOST_PP_CAT(NAME, _)); }                            \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                        \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE NAME(size_type _soa_impl_index) const {                                      \
        return *(BOOST_PP_CAT(NAME, _) + _soa_impl_index);                                                             \
      }                                                                                                                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      SOA_HOST_DEVICE SOA_INLINE CPP_TYPE::Scalar const* NAME() const { return BOOST_PP_CAT(NAME, _); }                \
      SOA_HOST_DEVICE SOA_INLINE size_type BOOST_PP_CAT(NAME, Stride)() { return BOOST_PP_CAT(NAME, Stride_); }        \
  )
// clang-format on

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                     \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME))

/**
 * SoA member ROOT streamer read (column pointers).
 */
// clang-format off
#define _STREAMER_READ_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                          \
    _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
      /* Scalar */                                                                                                     \
      memcpy(BOOST_PP_CAT(NAME, _), onfile.BOOST_PP_CAT(NAME, _), sizeof(CPP_TYPE));                                   \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      memcpy(BOOST_PP_CAT(NAME, _), onfile.BOOST_PP_CAT(NAME, _), sizeof(CPP_TYPE) * onfile.elements_);                \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      memcpy(BOOST_PP_CAT(NAME, _), onfile.BOOST_PP_CAT(NAME, _),                                                      \
        sizeof(CPP_TYPE::Scalar) * BOOST_PP_CAT(NAME, ElementsWithPadding_));                                          \
	)
// clang-format on

#define _STREAMER_READ_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                  \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_STREAMER_READ_SOA_DATA_MEMBER_IMPL TYPE_NAME))

// clang-format off
#define _DECLARE_SOA_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
_SWITCH_ON_TYPE(VALUE_TYPE,                                                                                            \
  /* Scalar */                                                                                                         \
  CPP_TYPE* BOOST_PP_CAT(NAME, _) EDM_REFLEX_SIZE(scalar_) = nullptr;                                                  \
  ,                                                                                                                    \
  /* Column */                                                                                                         \
  CPP_TYPE * BOOST_PP_CAT(NAME, _) EDM_REFLEX_SIZE(elements_) = nullptr;                                               \
  ,                                                                                                                    \
  /* Eigen column */                                                                                                   \
  size_type BOOST_PP_CAT(NAME, ElementsWithPadding_) = 0; /* For ROOT serialization */                                 \
  CPP_TYPE::Scalar * BOOST_PP_CAT(NAME, _) EDM_REFLEX_SIZE(BOOST_PP_CAT(NAME, ElementsWithPadding_)) = nullptr;        \
  byte_size_type BOOST_PP_CAT(NAME, Stride_) = 0;                                                                      \
)
// clang-format on

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                        \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME))

// clang-format off
#define _COPY_VIEW_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)           \
  _SWITCH_ON_TYPE(VALUE_TYPE, /* Scalar */                                  \
                  memcpy(BOOST_PP_CAT(this->metadata().addressOf_, NAME)(), \
                         BOOST_PP_CAT(view.metadata().addressOf_, NAME)(),  \
                         sizeof(CPP_TYPE));                                 \
                  , /* Column */                                            \
                  memcpy(BOOST_PP_CAT(this->metadata().addressOf_, NAME)(), \
                         BOOST_PP_CAT(view.metadata().addressOf_, NAME)(),  \
                         view.metadata().size() * sizeof(CPP_TYPE));        \
                  , /* Eigen column */                                      \
                  memcpy(BOOST_PP_CAT(this->metadata().addressOf_, NAME)(), \
                         BOOST_PP_CAT(view.metadata().addressOf_, NAME)(),  \
                         BOOST_PP_CAT(NAME, ElementsWithPadding_) * sizeof(CPP_TYPE::Scalar));)
// clang-format on

#define _COPY_VIEW_COLUMNS(R, DATA, TYPE_NAME)                                              \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_COPY_VIEW_COLUMNS_IMPL TYPE_NAME))

// -------- MACROS FOR GENERATING SOA VIEWS --------

/**
 * Member types aliasing for referencing by name
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS, CAST)                                   \
  using BOOST_PP_CAT(ParametersTypeOf_, NAME) =                                                                        \
      typename TypeOf_Layout::Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME);                                         \
  using BOOST_PP_CAT(TypeOf_, NAME) = CPP_TYPE;                                                                        \
  constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, NAME) =                                         \
      TypeOf_Layout::Metadata::BOOST_PP_CAT(ColumnTypeOf_, NAME);                                                      \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
  const auto BOOST_PP_CAT(parametersOf_, NAME)() const {                                                               \
    return CAST(parent_.BOOST_PP_CAT(NAME, Parameters_));                                                              \
  };
// clang-format on

// DATA should be a function used to convert
//   parent_.LOCAL_NAME ## Parameters_
// to
//   ParametersTypeOf_ ## LOCAL_NAME                (for a View)
// or
//   ParametersTypeOf_ ## LOCAL_NAME :: ConstType   (for a ConstView)
// or empty, if no conversion is necessary.
#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS(R, DATA, TYPE_NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA)))

/**
 * Member type const pointers for referencing by name
 */
#define _DECLARE_VIEW_MEMBER_CONST_POINTERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  SOA_HOST_DEVICE SOA_INLINE auto const* BOOST_PP_CAT(addressOf_, NAME)() const {  \
    return parent_.BOOST_PP_CAT(NAME, Parameters_).addr_;                          \
  };

#define _DECLARE_VIEW_MEMBER_CONST_POINTERS(R, DATA, TYPE_NAME)                             \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_CONST_POINTERS_IMPL TYPE_NAME))

/**
 * Assign the value of the records to the column parameters.
 */
#define _STRUCT_ELEMENT_INITIALIZERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (BOOST_PP_CAT(NAME, _){parent_.metadata().BOOST_PP_CAT(parametersOf_, NAME)()})

#define _STRUCT_ELEMENT_INITIALIZERS(R, DATA, TYPE_NAME)                                    \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_STRUCT_ELEMENT_INITIALIZERS_IMPL TYPE_NAME))

/**
 * Generator of accessors for (const) view Metarecords subclass.
 */
#define _CONST_ACCESSORS_STRUCT_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                        \
  cms::soa::ConstTuple<typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)>::Type NAME() const { \
    return std::make_tuple(BOOST_PP_CAT(NAME, _), parent_.elements_);                                 \
  }

#define _CONST_ACCESSORS_STRUCT_MEMBERS(R, DATA, TYPE_NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_CONST_ACCESSORS_STRUCT_MEMBERS_IMPL TYPE_NAME))

/**
 * Generator of members for (const) view Metarecords subclass.
 */
#define _DECLARE_STRUCT_CONST_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)::ConstType BOOST_PP_CAT(NAME, _);

#define _DECLARE_STRUCT_CONST_DATA_MEMBER(R, srcDATA, TYPE_NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_STRUCT_CONST_DATA_MEMBER_IMPL TYPE_NAME))

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                       \
  (BOOST_PP_CAT(NAME, Parameters_)([&]() -> auto {                                                                     \
    auto params = layout.metadata().BOOST_PP_CAT(parametersOf_, NAME)();                                               \
    if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                              \
      if (reinterpret_cast<intptr_t>(params.addr_) % alignment)                                                        \
        throw std::runtime_error("In constructor by layout: misaligned column: " #NAME);                               \
    return params;                                                                                                     \
  }()))
// clang-format on

#define _DECLARE_VIEW_MEMBER_INITIALIZERS(R, DATA, TYPE_NAME)                               \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL TYPE_NAME))

/**
 * Generator of parameters for constructor by column.
 */
#define _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS, DATA) \
  (typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, NAME)::TupleOrPointerType NAME)

#define _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS(R, DATA, TYPE_NAME)          \
  BOOST_PP_IF(                                                                      \
      BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
      BOOST_PP_EMPTY(),                                                             \
      BOOST_PP_EXPAND(_DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA)))

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                              \
  (                                                                                                                    \
    BOOST_PP_CAT(NAME, Parameters_)([&]() -> auto {                                                                    \
      if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                            \
        if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::checkAlignment(NAME, alignment))                         \
          throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                             \
      return NAME;                                                                                                     \
    }())                                                                                                               \
  )
// clang-format on

#define _DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN(R, DATA, TYPE_NAME)                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN_IMPL TYPE_NAME))

/**
 * Generator of parameters for (const) view Metarecords subclass.
 */
#define _DECLARE_CONST_VIEW_CONSTRUCTOR_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (cms::soa::ConstTuple<typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)>::Type NAME)

#define _DECLARE_CONST_VIEW_CONSTRUCTOR_COLUMNS(R, DATA, TYPE_NAME)                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_CONSTRUCTOR_COLUMNS_IMPL TYPE_NAME))

// clang-format off
#define _INITIALIZE_CONST_VIEW_PARAMETERS_AND_SIZE_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                              \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar */                                                                                                     \
        if (not readyToSet) {                                                                                          \
          elements_ = std::get<1>(NAME);                                                                               \
          readyToSet = true;                                                                                           \
        }                                                                                                              \
        BOOST_PP_CAT(NAME, Parameters_) = [&]() -> auto {                                                              \
          if (elements_ != std::get<1>(NAME))                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                        \
            if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::                                                     \
              checkAlignment(std::get<0>(NAME).tupleOrPointer(), alignment))                                           \
                throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                       \
          return std::get<0>(NAME);                                                                                    \
            }();                                                                                                       \
        ,                                                                                                              \
      /* Column */                                                                                                     \
        if (not readyToSet) {                                                                                          \
          elements_ = std::get<1>(NAME);                                                                               \
          readyToSet = true;                                                                                           \
        }                                                                                                              \
        BOOST_PP_CAT(NAME, Parameters_) = [&]() -> auto {                                                              \
          if (elements_ != std::get<1>(NAME))                                                                          \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: number of elements not equal for every column: "                     \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                        \
            if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::                                                     \
              checkAlignment(std::get<0>(NAME).tupleOrPointer(), alignment))                                           \
                throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                       \
          return std::get<0>(NAME);                                                                                    \
            }();                                                                                                       \
        ,                                                                                                              \
      /* Eigen column */                                                                                               \
        if (not readyToSet) {                                                                                          \
          elements_ = std::get<1>(NAME);                                                                               \
          readyToSet = true;                                                                                           \
        }                                                                                                              \
        BOOST_PP_CAT(NAME, Parameters_) = [&]() -> auto {                                                              \
          if (cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment)                                     \
                    / sizeof(CPP_TYPE::Scalar) != std::get<0>(NAME).stride_) {                                         \
            throw std::runtime_error(                                                                                  \
              "In constructor by column pointers: stride not equal between eigen columns: "                            \
              BOOST_PP_STRINGIZE(NAME));                                                                               \
          }                                                                                                            \
          if (elements_ != std::get<1>(NAME))                                                                          \
          throw std::runtime_error(                                                                                    \
            "In constructor by column pointers: number of elements not equal for every column: "                       \
            BOOST_PP_STRINGIZE(NAME));                                                                                 \
          if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                        \
            if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::                                                     \
              checkAlignment(std::get<0>(NAME).tupleOrPointer(), alignment))                                           \
                throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                       \
          return std::get<0>(NAME);                                                                                    \
          }();                                                                                                         \
  )
// clang-format on

#define _INITIALIZE_CONST_VIEW_PARAMETERS_AND_SIZE(R, DATA, TYPE_NAME)                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_INITIALIZE_CONST_VIEW_PARAMETERS_AND_SIZE_IMPL TYPE_NAME))

/**
 * Generator of view member list.
 */
#define _DECLARE_VIEW_OTHER_MEMBER_LIST_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (cms::soa::const_cast_SoAParametersImpl(other.BOOST_PP_CAT(NAME, Parameters_)).tupleOrPointer())

#define _DECLARE_VIEW_OTHER_MEMBER_LIST(R, DATA, TYPE_NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_OTHER_MEMBER_LIST_IMPL TYPE_NAME))

/**
 * Generator of parameters for (const) element subclass (expanded comma separated).
 */
#define _DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (const typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, NAME)::ConstType NAME)

#define _DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME)                           \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG_IMPL TYPE_NAME))

/**
 * Generator of member initialization for constructor of element subclass
 */
#define _DECLARE_VIEW_CONST_ELEM_MEMBER_INIT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS, DATA) \
  (BOOST_PP_CAT(NAME, _)(DATA, NAME))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_VIEW_CONST_ELEM_MEMBER_INIT(R, DATA, TYPE_NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA)))

/**
 * Declaration of the members accessors of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                    \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
      const typename cms::soa::SoAConstValue_ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template         \
              DataType<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::template                                       \
                  Alignment<conditionalAlignment>::template ConstValue<restrictQualify>::RefToConst                    \
      NAME() const {                                                                                                   \
    return BOOST_PP_CAT(NAME, _)();                                                                                    \
  }
// clang-format on

#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

/**
 * Declaration of the private members of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                               \
  const cms::soa::ConstValueTraitsFromC<typename cms::soa::SoAConstValue_ColumnType<                                  \
      BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template                                                          \
          DataType<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::template                                          \
              Alignment<conditionalAlignment>::template ConstValue<restrictQualify>>                                  \
      BOOST_PP_CAT(NAME, _);
// clang-format on

#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME)                        \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME))

/**
 * Parameters passed to const element subclass constructor in operator[]
 */
#define _DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) (BOOST_PP_CAT(NAME, Parameters_))

#define _DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME)                         \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME))

/**
 * Direct access to column pointer (const) and indexed access.
 */
// clang-format off
#define _DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                      \
_SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
  /* Scalar */                                                                                                       \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                      \
  NAME() const {                                                                                                     \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(NAME, Parameters_))();                      \
  }                                                                                                                  \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                        \
  NAME(size_type _soa_impl_index) const {                                                                            \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
      if (_soa_impl_index >= elements_ or _soa_impl_index < 0)                                                       \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #NAME "(size_type index)",                           \
          _soa_impl_index, elements_)                                                                                \
    }                                                                                                                \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(NAME, Parameters_))(_soa_impl_index);       \
  }                                                                                                                  \
  ,                                                                                                                  \
  /* Column */                                                                                                       \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                      \
  NAME() const {                                                                                                     \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(NAME, Parameters_), elements_)();           \
  }                                                                                                                  \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                        \
  NAME(size_type _soa_impl_index) const {                                                                            \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
      if (_soa_impl_index >= elements_ or _soa_impl_index < 0)                                                       \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #NAME "(size_type index)",                           \
          _soa_impl_index, elements_)                                                                                \
    }                                                                                                                \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(NAME, Parameters_),                         \
                    elements_)(_soa_impl_index);                                                                     \
  }                                                                                                                  \
  ,                                                                                                                  \
  /* Eigen column */                                                                                                 \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                      \
  NAME() const {                                                                                                     \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(NAME, Parameters_),                         \
                        cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment) /                       \
                            sizeof(CPP_TYPE::Scalar) * CPP_TYPE::RowsAtCompileTime *                                 \
                                CPP_TYPE::ColsAtCompileTime)();                                                      \
  }                                                                                                                  \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                        \
  NAME(size_type _soa_impl_index) const {                                                                            \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
      if (_soa_impl_index >= elements_ or _soa_impl_index < 0)                                                       \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #NAME "(size_type index)",                           \
          _soa_impl_index, elements_)                                                                                \
    }                                                                                                                \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(NAME, Parameters_),                         \
                        cms::soa::alignSize(elements_ * sizeof(CPP_TYPE::Scalar), alignment) /                       \
                            sizeof(CPP_TYPE::Scalar) * CPP_TYPE::RowsAtCompileTime *                                 \
                                CPP_TYPE::ColsAtCompileTime)(_soa_impl_index);                                       \
  }                                                                                                                  \
)
// clang-format on

#define _DECLARE_VIEW_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL TYPE_NAME))

/**
 * Const SoA class member declaration (column pointers and parameters).
 */
#define _DECLARE_CONST_VIEW_SOA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, NAME)::ConstType BOOST_PP_CAT(NAME, Parameters_);

#define _DECLARE_CONST_VIEW_SOA_MEMBER(R, DATA, TYPE_NAME)                                  \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_SOA_MEMBER_IMPL TYPE_NAME))

/**
 * Member type pointers for referencing by name
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_POINTERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                         \
  SOA_HOST_DEVICE SOA_INLINE auto* BOOST_PP_CAT(addressOf_, NAME)() {                                                \
    return BOOST_PP_CAT(parametersOf_, NAME)().addr_;                                                                \
  };
// clang-format on

#define _DECLARE_VIEW_MEMBER_POINTERS(R, DATA, TYPE_NAME)                                   \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_POINTERS_IMPL TYPE_NAME))

/**
 * Generator of accessors for (const) view Metarecords subclass.
 */
#define _ACCESSORS_STRUCT_MEMBERS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                         \
  cms::soa::Tuple<typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)>::Type NAME() const { \
    return std::make_tuple(BOOST_PP_CAT(NAME, _), parent_.elements_);                            \
  }

#define _ACCESSORS_STRUCT_MEMBERS(R, DATA, TYPE_NAME)                                       \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_ACCESSORS_STRUCT_MEMBERS_IMPL TYPE_NAME))

/**
 * Generator of members for view Metarecords subclass.
 */
#define _DECLARE_STRUCT_DATA_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME) BOOST_PP_CAT(NAME, _);

#define _DECLARE_STRUCT_DATA_MEMBER(R, DATA, TYPE_NAME)                                     \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_STRUCT_DATA_MEMBER_IMPL TYPE_NAME))

/**
 * Generator of view member list.
 */
#define _DECLARE_VIEW_MEMBER_LIST_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) (NAME)

#define _DECLARE_VIEW_MEMBER_LIST(R, DATA, TYPE_NAME)                                       \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_LIST_IMPL TYPE_NAME))

/**
 * Generator of parameters for view Metarecords subclass.
 */
#define _DECLARE_VIEW_CONSTRUCTOR_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (cms::soa::Tuple<typename Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)>::Type NAME)

#define _DECLARE_VIEW_CONSTRUCTOR_COLUMNS(R, DATA, TYPE_NAME)                               \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_CONSTRUCTOR_COLUMNS_IMPL TYPE_NAME))

// clang-format off
#define _INITIALIZE_VIEW_PARAMETERS_AND_SIZE_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                  \
_SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
    /* Scalar */                                                                                                     \
      if (not readyToSet) {                                                                                          \
        base_type::elements_ = std::get<1>(NAME);                                                                    \
        readyToSet = true;                                                                                           \
      }                                                                                                              \
      base_type::BOOST_PP_CAT(NAME, Parameters_) = [&]() -> auto {                                                   \
        if (base_type::elements_ != std::get<1>(NAME))                                                               \
          throw std::runtime_error(                                                                                  \
            "In constructor by column pointers: number of elements not equal for every column: "                     \
            BOOST_PP_STRINGIZE(NAME));                                                                               \
        if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                        \
          if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::                                                     \
            checkAlignment(std::get<0>(NAME).tupleOrPointer(), alignment))                                           \
              throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                       \
        return std::get<0>(NAME);                                                                                    \
          }();                                                                                                       \
      ,                                                                                                              \
    /* Column */                                                                                                     \
      if (not readyToSet) {                                                                                          \
        base_type::elements_ = std::get<1>(NAME);                                                                    \
        readyToSet = true;                                                                                           \
      }                                                                                                              \
      base_type::BOOST_PP_CAT(NAME, Parameters_) = [&]() -> auto {                                                   \
        if (base_type::elements_ != std::get<1>(NAME))                                                               \
          throw std::runtime_error(                                                                                  \
            "In constructor by column pointers: number of elements not equal for every column: "                     \
            BOOST_PP_STRINGIZE(NAME));                                                                               \
        if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                        \
          if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::                                                     \
            checkAlignment(std::get<0>(NAME).tupleOrPointer(), alignment))                                           \
              throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                       \
        return std::get<0>(NAME);                                                                                    \
          }();                                                                                                       \
      ,                                                                                                              \
    /* Eigen column */                                                                                               \
      if (not readyToSet) {                                                                                          \
        base_type::elements_ = std::get<1>(NAME);                                                                    \
        readyToSet = true;                                                                                           \
      }                                                                                                              \
      base_type::BOOST_PP_CAT(NAME, Parameters_) = [&]() -> auto {                                                   \
        if (cms::soa::alignSize(base_type::elements_ * sizeof(CPP_TYPE::Scalar), alignment)                          \
                  / sizeof(CPP_TYPE::Scalar) != std::get<0>(NAME).stride_) {                                         \
          throw std::runtime_error(                                                                                  \
            "In constructor by column pointers: stride not equal between eigen columns: "                            \
            BOOST_PP_STRINGIZE(NAME));                                                                               \
        }                                                                                                            \
        if (base_type::elements_ != std::get<1>(NAME))                                                               \
        throw std::runtime_error(                                                                                    \
          "In constructor by column pointers: number of elements not equal for every column: "                       \
          BOOST_PP_STRINGIZE(NAME));                                                                                 \
        if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                        \
          if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::                                                     \
            checkAlignment(std::get<0>(NAME).tupleOrPointer(), alignment))                                           \
              throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                       \
        return std::get<0>(NAME);                                                                                    \
        }();                                                                                                         \
  )
// clang-format on

#define _INITIALIZE_VIEW_PARAMETERS_AND_SIZE(R, DATA, TYPE_NAME)                            \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_INITIALIZE_VIEW_PARAMETERS_AND_SIZE_IMPL TYPE_NAME))

#define _DECLARE_CONSTRUCTOR_CONST_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME)::ConstType NAME)

#define _DECLARE_CONSTRUCTOR_CONST_COLUMNS(R, DATA, TYPE_NAME)                              \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONSTRUCTOR_CONST_COLUMNS_IMPL TYPE_NAME))

#define _DECLARE_CONSTRUCTOR_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (Metadata::BOOST_PP_CAT(ParametersTypeOf_, NAME) NAME)

#define _DECLARE_CONSTRUCTOR_COLUMNS(R, DATA, TYPE_NAME)                                    \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_CONSTRUCTOR_COLUMNS_IMPL TYPE_NAME))

#define _INITIALIZE_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) (NAME)

#define _INITIALIZE_COLUMNS(R, DATA, TYPE_NAME)                                             \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_INITIALIZE_COLUMNS_IMPL TYPE_NAME))

#define _INITIALIZE_CONST_COLUMNS_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) (BOOST_PP_CAT(NAME, Parameters_){NAME})

#define _INITIALIZE_CONST_COLUMNS(R, DATA, TYPE_NAME)                                       \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_INITIALIZE_CONST_COLUMNS_IMPL TYPE_NAME))

/**
 * Generator of parameters for (non-const) element subclass (expanded comma separated).
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, NAME) NAME)

#define _DECLARE_VIEW_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME)                                 \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL TYPE_NAME))

/**
 * Generator of element members initializer.
 */
#define _DECLARE_VIEW_ELEM_MEMBER_INIT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS, DATA) (NAME(DATA, NAME))

#define _DECLARE_VIEW_ELEM_MEMBER_INIT(R, DATA, TYPE_NAME)                                  \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_ELEM_MEMBER_INIT_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA)))

/**
 * Generator of the member-by-member copy operator of the element subclass.
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                   \
  if constexpr (Metadata::BOOST_PP_CAT(ColumnTypeOf_, NAME) != cms::soa::SoAColumnType::scalar) { \
    NAME() = _soa_impl_other.NAME();                                                              \
  }

#define _DECLARE_VIEW_ELEMENT_VALUE_COPY(R, DATA, TYPE_NAME)                                \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL TYPE_NAME))

/**
 * Assign the value of the view from the values in the value_element.
 */
// clang-format off
#define _TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                    \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
      /* Scalar (empty) */                                                                                           \
      ,                                                                                                              \
      /* Column */                                                                                                   \
      NAME() = _soa_impl_value.NAME;                                                                                 \
      ,                                                                                                              \
      /* Eigen column */                                                                                             \
      NAME() = _soa_impl_value.NAME;                                                                                 \
)
// clang-format on

#define _TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT(R, DATA, TYPE_NAME)                              \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT_IMPL TYPE_NAME))

/**
 * Declaration of the private members of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                    \
  typename cms::soa::SoAValue_ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template                      \
              DataType<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::template                                     \
                  Alignment<conditionalAlignment>::template Value<restrictQualify>                                   \
      NAME;
// clang-format on

#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME)                              \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME))

/**
 * Parameters passed to element subclass constructor in operator[]
 *
 * The use of const_cast (inside cms::soa::non_const_ptr) is safe because the constructor of a View binds only to
 * non-const arguments.
 */
#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS) \
  (cms::soa::const_cast_SoAParametersImpl(base_type::BOOST_PP_CAT(NAME, Parameters_)))

#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME)                               \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME))
/**
 * Direct access to column pointer and indexed access
 */
// clang-format off
#define _DECLARE_VIEW_SOA_ACCESSOR_IMPL(VALUE_TYPE, CPP_TYPE, NAME, ARGS)                                            \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                        \
  /* Scalar */                                                                                                       \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                      \
  NAME() {                                                                                                           \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                template RestrictQualifier<restrictQualify>(cms::soa::const_cast_SoAParametersImpl(                  \
                    base_type:: BOOST_PP_CAT(NAME, Parameters_)))();                                                 \
  }                                                                                                                  \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                        \
  NAME(size_type _soa_impl_index) {                                                                                  \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
      if (_soa_impl_index >= base_type::elements_ or _soa_impl_index < 0)                                            \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #NAME "(size_type index)",                           \
          _soa_impl_index, base_type::elements_)                                                                     \
    }                                                                                                                \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                template RestrictQualifier<restrictQualify>(cms::soa::const_cast_SoAParametersImpl(                  \
                    base_type:: BOOST_PP_CAT(NAME, Parameters_)))(_soa_impl_index);                                  \
  }                                                                                                                  \
  ,                                                                                                                  \
  /* Column */                                                                                                       \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                      \
  NAME() {                                                                                                           \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                template RestrictQualifier<restrictQualify>(cms::soa::const_cast_SoAParametersImpl(                  \
                    base_type:: BOOST_PP_CAT(NAME, Parameters_)), base_type::elements_)();                           \
  }                                                                                                                  \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                        \
  NAME(size_type _soa_impl_index) {                                                                                  \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
      if (_soa_impl_index >= base_type::elements_ or _soa_impl_index < 0)                                            \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #NAME "(size_type index)",                           \
          _soa_impl_index, base_type::elements_)                                                                     \
    }                                                                                                                \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                template RestrictQualifier<restrictQualify>(cms::soa::const_cast_SoAParametersImpl(                  \
                    base_type:: BOOST_PP_CAT(NAME, Parameters_)), base_type::elements_)(_soa_impl_index);            \
  }                                                                                                                  \
  ,                                                                                                                  \
  /* Eigen column */                                                                                                 \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                      \
  NAME() {                                                                                                           \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                template RestrictQualifier<restrictQualify>(cms::soa::const_cast_SoAParametersImpl(                  \
                    base_type:: BOOST_PP_CAT(NAME, Parameters_)),                                                    \
                        cms::soa::alignSize(base_type::elements_ * sizeof(CPP_TYPE::Scalar), alignment) /            \
                            sizeof(CPP_TYPE::Scalar) * CPP_TYPE::RowsAtCompileTime *                                 \
                                CPP_TYPE::ColsAtCompileTime)();                                                      \
  }                                                                                                                  \
  SOA_HOST_DEVICE SOA_INLINE                                                                                         \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                                  \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                        \
  NAME(size_type _soa_impl_index) {                                                                                  \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
      if (_soa_impl_index >= base_type::elements_ or _soa_impl_index < 0)                                            \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #NAME "(size_type index)",                           \
          _soa_impl_index, base_type::elements_)                                                                     \
    }                                                                                                                \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, NAME)>::                         \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, NAME)>::template AccessType<                       \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                      \
                template RestrictQualifier<restrictQualify>(cms::soa::const_cast_SoAParametersImpl(                  \
                    base_type:: BOOST_PP_CAT(NAME, Parameters_)),                                                    \
                        cms::soa::alignSize(base_type::elements_ * sizeof(CPP_TYPE::Scalar), alignment) /            \
                            sizeof(CPP_TYPE::Scalar) * CPP_TYPE::RowsAtCompileTime *                                 \
                                CPP_TYPE::ColsAtCompileTime)(_soa_impl_index);                                       \
  }                                                                                                                  \
)
// clang-format on

#define _DECLARE_VIEW_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                      \
  BOOST_PP_IF(BOOST_PP_GREATER(BOOST_PP_TUPLE_ELEM(0, TYPE_NAME), _VALUE_LAST_COLUMN_TYPE), \
              BOOST_PP_EMPTY(),                                                             \
              BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_ACCESSOR_IMPL TYPE_NAME))

#ifdef DEBUG
#define _DO_RANGECHECK true
#else
#define _DO_RANGECHECK false
#endif

/*
 * A macro defining a SoA layout (collection of scalars and columns of equal lengths)
 */
// clang-format off
#define GENERATE_SOA_LAYOUT(CLASS, ...)                                                                                \
  template <CMS_SOA_BYTE_SIZE_TYPE ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                                   \
            bool ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed>                                      \
  struct CLASS {                                                                                                       \
    /* these could be moved to an external type trait to free up the symbol names */                                   \
    using self_type = CLASS;                                                                                           \
    using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                       \
                                                                                                                       \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                            \
     * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid      \
     * up to compute capability 8.X.                                                                                   \
     */                                                                                                                \
    using size_type = cms::soa::size_type;                                                                             \
    using byte_size_type = cms::soa::byte_size_type;                                                                   \
    constexpr static byte_size_type defaultAlignment = 128;                                                            \
    constexpr static byte_size_type alignment = ALIGNMENT;                                                             \
    constexpr static bool alignmentEnforcement = ALIGNMENT_ENFORCEMENT;                                                \
    constexpr static byte_size_type conditionalAlignment =                                                             \
        alignmentEnforcement == cms::soa::AlignmentEnforcement::enforced ? alignment : 0;                              \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                          \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment>;                                 \
                                                                                                                       \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment>;                       \
                                                                                                                       \
    template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                            \
            bool VIEW_ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,                                 \
            bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Default,                                                \
            bool RANGE_CHECKING = cms::soa::RangeChecking::Default>                                                    \
    struct ViewTemplateFreeParams;                                                                                     \
                                                                                                                       \
    /* dump the SoA internal structure */                                                                              \
    SOA_HOST_ONLY                                                                                                      \
    void soaToStreamInternal(std::ostream & _soa_impl_os) const {                                                      \
      _soa_impl_os << #CLASS "(" << elements_ << " elements, byte alignement= " << alignment << ", @"<< mem_ <<"): "   \
         << std::endl;                                                                                                 \
      _soa_impl_os << "  sizeof(" #CLASS "): " << sizeof(CLASS) << std::endl;                                          \
      byte_size_type _soa_impl_offset = 0;                                                                             \
      _ITERATE_ON_ALL(_DECLARE_SOA_STREAM_INFO, ~, __VA_ARGS__)                                                        \
      _soa_impl_os << "Final offset = " << _soa_impl_offset << " computeDataSize(...): " << computeDataSize(elements_) \
              << std::endl;                                                                                            \
      _soa_impl_os << std::endl;                                                                                       \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper function used by caller to externally allocate the storage */                                            \
    static constexpr byte_size_type computeDataSize(size_type elements) {                                              \
      byte_size_type _soa_impl_ret = 0;                                                                                \
      _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                         \
      return _soa_impl_ret;                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    /**                                                                                                                \
     * Helper/friend class allowing SoA introspection.                                                                 \
     */                                                                                                                \
    struct Metadata {                                                                                                  \
      friend CLASS;                                                                                                    \
      SOA_HOST_DEVICE SOA_INLINE size_type size() const { return parent_.elements_; }                                  \
      SOA_HOST_DEVICE SOA_INLINE byte_size_type byteSize() const { return parent_.byteSize_; }                         \
      SOA_HOST_DEVICE SOA_INLINE byte_size_type alignment() const { return CLASS::alignment; }                         \
      SOA_HOST_DEVICE SOA_INLINE std::byte* data() { return parent_.mem_; }                                            \
      SOA_HOST_DEVICE SOA_INLINE const std::byte* data() const { return parent_.mem_; }                                \
      SOA_HOST_DEVICE SOA_INLINE std::byte* nextByte() const { return parent_.mem_ + parent_.byteSize_; }              \
      SOA_HOST_DEVICE SOA_INLINE CLASS cloneToNewAddress(std::byte* _soa_impl_addr) const {                            \
        return CLASS(_soa_impl_addr, parent_.elements_);                                                               \
      }                                                                                                                \
                                                                                                                       \
      _ITERATE_ON_ALL(_DEFINE_METADATA_MEMBERS, ~, __VA_ARGS__)                                                        \
                                                                                                                       \
      struct value_element {                                                                                           \
        SOA_HOST_DEVICE SOA_INLINE value_element                                                                       \
          BOOST_PP_IF(                                                                                                 \
            BOOST_PP_SEQ_SIZE(_ITERATE_ON_ALL(_VALUE_ELEMENT_CTOR_ARGS, ~, __VA_ARGS__) ),                             \
            (_ITERATE_ON_ALL_COMMA(_VALUE_ELEMENT_CTOR_ARGS, ~, __VA_ARGS__)):,                                        \
            ())                                                                                                        \
          BOOST_PP_TUPLE_ENUM(BOOST_PP_IF(                                                                             \
            BOOST_PP_SEQ_SIZE(_ITERATE_ON_ALL(_VALUE_ELEMENT_CTOR_ARGS, ~, __VA_ARGS__)),                              \
            BOOST_PP_SEQ_TO_TUPLE(_ITERATE_ON_ALL(_VALUE_ELEMENT_INITIALIZERS, ~, __VA_ARGS__)),                       \
            ()                                                                                                         \
          )                                                                                                            \
        )                                                                                                              \
        {}                                                                                                             \
                                                                                                                       \
        _ITERATE_ON_ALL(_DEFINE_VALUE_ELEMENT_MEMBERS, ~, __VA_ARGS__)                                                 \
      };                                                                                                               \
                                                                                                                       \
      Metadata& operator=(const Metadata&) = delete;                                                                   \
      Metadata(const Metadata&) = delete;                                                                              \
                                                                                                                       \
    private:                                                                                                           \
      SOA_HOST_DEVICE SOA_INLINE Metadata(const CLASS& _soa_impl_parent) : parent_(_soa_impl_parent) {}                \
      const CLASS& parent_;                                                                                            \
      using ParentClass = CLASS;                                                                                       \
    };                                                                                                                 \
                                                                                                                       \
    friend Metadata;                                                                                                   \
                                                                                                                       \
    SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                             \
    SOA_HOST_DEVICE SOA_INLINE Metadata metadata() { return Metadata(*this); }                                         \
                                                                                                                       \
    template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                   \
              bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                         \
              bool RESTRICT_QUALIFY,                                                                                   \
              bool RANGE_CHECKING>                                                                                     \
    struct ConstViewTemplateFreeParams {                                                                               \
      /* these could be moved to an external type trait to free up the symbol names */                                 \
      using self_type = ConstViewTemplateFreeParams;                                                                   \
      using BOOST_PP_CAT(CLASS, _parametrized) = CLASS<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT>;                    \
      using size_type = cms::soa::size_type;                                                                           \
      using byte_size_type = cms::soa::byte_size_type;                                                                 \
      using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                     \
                                                                                                                       \
      template <CMS_SOA_BYTE_SIZE_TYPE, bool, bool, bool>                                                              \
      friend struct ViewTemplateFreeParams;                                                                            \
                                                                                                                       \
      template <CMS_SOA_BYTE_SIZE_TYPE, bool, bool, bool>                                                              \
      friend struct ConstViewTemplateFreeParams;                                                                       \
                                                                                                                       \
      /* For CUDA applications, we align to the 128 bytes of the cache lines.                                          \
        * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid   \
        * up to compute capability 8.X.                                                                                \
        */                                                                                                             \
      constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;                         \
      constexpr static byte_size_type alignment = VIEW_ALIGNMENT;                                                      \
      constexpr static bool alignmentEnforcement = VIEW_ALIGNMENT_ENFORCEMENT;                                         \
      constexpr static byte_size_type conditionalAlignment =                                                           \
          alignmentEnforcement == AlignmentEnforcement::enforced ? alignment : 0;                                      \
      constexpr static bool restrictQualify = RESTRICT_QUALIFY;                                                        \
      constexpr static bool rangeChecking = RANGE_CHECKING;                                                            \
                                                                                                                       \
      /**                                                                                                              \
       * Helper/friend class allowing SoA introspection.                                                               \
       */                                                                                                              \
      struct Metadata {                                                                                                \
        friend ConstViewTemplateFreeParams;                                                                            \
        SOA_HOST_DEVICE SOA_INLINE size_type size() const { return parent_.elements_; }                                \
        /* Alias layout to name-derived identifyer to allow simpler definitions */                                     \
        using TypeOf_Layout = BOOST_PP_CAT(CLASS, _parametrized);                                                      \
                                                                                                                       \
        /* Alias member types to name-derived identifyer to allow simpler definitions */                               \
        _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, BOOST_PP_EMPTY(), __VA_ARGS__)                                \
        _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_CONST_POINTERS, ~, __VA_ARGS__)                                           \
                                                                                                                       \
        /* Forbid copying to avoid const correctness evasion */                                                        \
        Metadata& operator=(const Metadata&) = delete;                                                                 \
        Metadata(const Metadata&) = delete;                                                                            \
                                                                                                                       \
      private:                                                                                                         \
        SOA_HOST_DEVICE SOA_INLINE Metadata(const ConstViewTemplateFreeParams& _soa_impl_parent)                       \
        : parent_(_soa_impl_parent) {}                                                                                 \
        const ConstViewTemplateFreeParams& parent_;                                                                    \
      };                                                                                                               \
                                                                                                                       \
      friend Metadata;                                                                                                 \
                                                                                                                       \
      /**                                                                                                              \
      * Helper/friend class allowing access to size from columns.                                                      \
      */                                                                                                               \
      struct Metarecords {                                                                                             \
        friend ConstViewTemplateFreeParams;                                                                            \
		SOA_HOST_DEVICE SOA_INLINE\
        Metarecords(const ConstViewTemplateFreeParams& _soa_impl_parent) :                                             \
                    parent_(_soa_impl_parent), _ITERATE_ON_ALL_COMMA(_STRUCT_ELEMENT_INITIALIZERS, ~, __VA_ARGS__) {}  \
        _ITERATE_ON_ALL(_CONST_ACCESSORS_STRUCT_MEMBERS, ~, __VA_ARGS__)                                               \
        private:                                                                                                       \
          const ConstViewTemplateFreeParams& parent_;                                                                  \
          _ITERATE_ON_ALL(_DECLARE_STRUCT_CONST_DATA_MEMBER, ~, __VA_ARGS__)                                           \
      };                                                                                                               \
      SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                           \
      SOA_HOST_DEVICE SOA_INLINE const Metarecords records() const { return Metarecords(*this); }                      \
                                                                                                                       \
      /* Trivial constuctor */                                                                                         \
      ConstViewTemplateFreeParams() = default;                                                                         \
                                                                                                                       \
      /* Constructor relying the layout */                                                                             \
      SOA_HOST_ONLY ConstViewTemplateFreeParams(const Metadata::TypeOf_Layout& layout)                                 \
      : elements_(layout.metadata().size()),                                                                           \
        _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS, ~, __VA_ARGS__) {}                                    \
                                                                                                                       \
      /* Constructor relying on individually provided column structs */                                                \
      SOA_HOST_ONLY ConstViewTemplateFreeParams(_ITERATE_ON_ALL_COMMA(                                                 \
                _DECLARE_CONST_VIEW_CONSTRUCTOR_COLUMNS, ~, __VA_ARGS__)) {                                            \
        bool readyToSet = false;                                                                                       \
        _ITERATE_ON_ALL(_INITIALIZE_CONST_VIEW_PARAMETERS_AND_SIZE, ~, __VA_ARGS__)                                    \
      }                                                                                                                \
                                                                                                                       \
      /* Copiable */                                                                                                   \
      ConstViewTemplateFreeParams(ConstViewTemplateFreeParams const&) = default;                                       \
      ConstViewTemplateFreeParams& operator=(ConstViewTemplateFreeParams const&) = default;                            \
                                                                                                                       \
      /* Copy constructor for other parameters */                                                                      \
      template <CMS_SOA_BYTE_SIZE_TYPE OTHER_VIEW_ALIGNMENT,                                                           \
                bool OTHER_VIEW_ALIGNMENT_ENFORCEMENT,                                                                 \
                bool OTHER_RESTRICT_QUALIFY,                                                                           \
                bool OTHER_RANGE_CHECKING>                                                                             \
      ConstViewTemplateFreeParams(ConstViewTemplateFreeParams<OTHER_VIEW_ALIGNMENT,                                    \
        OTHER_VIEW_ALIGNMENT_ENFORCEMENT, OTHER_RESTRICT_QUALIFY, OTHER_RANGE_CHECKING> const& other)                  \
        : ConstViewTemplateFreeParams{other.elements_,                                                                 \
            _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_OTHER_MEMBER_LIST, BOOST_PP_EMPTY(), __VA_ARGS__)                      \
          } {}                                                                                                         \
                                                                                                                       \
      SOA_HOST_DEVICE                                                                                                  \
      ConstViewTemplateFreeParams(size_type elems,                                                                     \
        _ITERATE_ON_ALL_COMMA(_DECLARE_CONSTRUCTOR_CONST_COLUMNS, ~, __VA_ARGS__)) :                                   \
        elements_{elems}, _ITERATE_ON_ALL_COMMA(_INITIALIZE_CONST_COLUMNS, ~, __VA_ARGS__) { }                         \
                                                                                                                       \
      /* Copy operator for other parameters */                                                                         \
      template <CMS_SOA_BYTE_SIZE_TYPE OTHER_VIEW_ALIGNMENT,                                                           \
          bool OTHER_VIEW_ALIGNMENT_ENFORCEMENT,                                                                       \
          bool OTHER_RESTRICT_QUALIFY,                                                                                 \
          bool OTHER_RANGE_CHECKING>                                                                                   \
      ConstViewTemplateFreeParams& operator=(ConstViewTemplateFreeParams<OTHER_VIEW_ALIGNMENT,                         \
          OTHER_VIEW_ALIGNMENT_ENFORCEMENT, OTHER_RESTRICT_QUALIFY, OTHER_RANGE_CHECKING> const& other)                \
          { *this = other; }                                                                                           \
                                                                                                                       \
      /* Movable */                                                                                                    \
      ConstViewTemplateFreeParams(ConstViewTemplateFreeParams &&) = default;                                           \
      ConstViewTemplateFreeParams& operator=(ConstViewTemplateFreeParams &&) = default;                                \
                                                                                                                       \
      /* Trivial destuctor */                                                                                          \
      ~ConstViewTemplateFreeParams() = default;                                                                        \
                                                                                                                       \
      /* AoS-like accessor (const) */                                                                                  \
      struct const_element {                                                                                           \
        SOA_HOST_DEVICE SOA_INLINE                                                                                     \
        const_element(size_type _soa_impl_index, /* Declare parameters */                                              \
                      _ITERATE_ON_ALL_COMMA(_DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG, ~, __VA_ARGS__))                    \
                      : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT, _soa_impl_index, __VA_ARGS__) {}   \
        _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_ACCESSOR, ~, __VA_ARGS__)                                          \
                                                                                                                       \
        ENUM_IF_VALID(_ITERATE_ON_ALL(GENERATE_CONST_METHODS, ~, __VA_ARGS__))                                         \
                                                                                                                       \
        private:                                                                                                       \
        _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER, ~, __VA_ARGS__)                                      \
      };                                                                                                               \
                                                                                                                       \
        SOA_HOST_DEVICE SOA_INLINE                                                                                     \
        const_element operator[](size_type _soa_impl_index) const {                                                    \
          if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                           \
            if (_soa_impl_index >= elements_ or _soa_impl_index < 0)                                                   \
              SOA_THROW_OUT_OF_RANGE("Out of range index in ConstViewTemplateFreeParams " #CLASS "::operator[]",       \
                _soa_impl_index, elements_)                                                                            \
          }                                                                                                            \
          return const_element{                                                                                        \
            _soa_impl_index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL, ~, __VA_ARGS__)            \
          };                                                                                                           \
        }                                                                                                              \
                                                                                                                       \
        /* const accessors */                                                                                          \
        _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_CONST_ACCESSOR, ~, __VA_ARGS__)                                              \
                                                                                                                       \
        /* dump the SoA internal structure */                                                                          \
        template <typename T>                                                                                          \
        SOA_HOST_ONLY friend void dump();                                                                              \
                                                                                                                       \
        private:                                                                                                       \
          size_type elements_ = 0;                                                                                     \
          _ITERATE_ON_ALL(_DECLARE_CONST_VIEW_SOA_MEMBER, ~, __VA_ARGS__)                                              \
      };                                                                                                               \
                                                                                                                       \
    template <bool RESTRICT_QUALIFY, bool RANGE_CHECKING>                                                              \
    using ConstViewTemplate = ConstViewTemplateFreeParams<ALIGNMENT, ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY,          \
      RANGE_CHECKING>;                                                                                                 \
                                                                                                                       \
    using ConstView = ConstViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::Default>;         \
                                                                                                                       \
    template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                   \
              bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                         \
              bool RESTRICT_QUALIFY,                                                                                   \
              bool RANGE_CHECKING>                                                                                     \
      struct ViewTemplateFreeParams                                                                                    \
      : public ConstViewTemplateFreeParams<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT,                                 \
                                           RESTRICT_QUALIFY, RANGE_CHECKING> {                                         \
      /* these could be moved to an external type trait to free up the symbol names */                                 \
      using self_type = ViewTemplateFreeParams;                                                                        \
      using base_type = ConstViewTemplateFreeParams<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT,                        \
                                                    RESTRICT_QUALIFY, RANGE_CHECKING>;                                 \
      using BOOST_PP_CAT(CLASS, _parametrized) = CLASS<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT>;                    \
      using size_type = cms::soa::size_type;                                                                           \
      using byte_size_type = cms::soa::byte_size_type;                                                                 \
      using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                     \
                                                                                                                       \
      /* For CUDA applications, we align to the 128 bytes of the cache lines.                                          \
       * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid    \
       * up to compute capability 8.X.                                                                                 \
       */                                                                                                              \
      constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;                         \
      constexpr static byte_size_type alignment = VIEW_ALIGNMENT;                                                      \
      constexpr static bool alignmentEnforcement = VIEW_ALIGNMENT_ENFORCEMENT;                                         \
      constexpr static byte_size_type conditionalAlignment =                                                           \
          alignmentEnforcement == AlignmentEnforcement::enforced ? alignment : 0;                                      \
      constexpr static bool restrictQualify = RESTRICT_QUALIFY;                                                        \
      constexpr static bool rangeChecking = RANGE_CHECKING;                                                            \
                                                                                                                       \
      template <CMS_SOA_BYTE_SIZE_TYPE, bool, bool, bool>                                                              \
      friend struct ViewTemplateFreeParams;                                                                            \
                                                                                                                       \
      /**                                                                                                              \
       * Helper/friend class allowing SoA introspection.                                                               \
       */                                                                                                              \
      struct Metadata {                                                                                                \
        friend ViewTemplateFreeParams;                                                                                 \
        SOA_HOST_DEVICE SOA_INLINE size_type size() const { return parent_.elements_; }                                \
        /* Alias layout to name-derived identifyer to allow simpler definitions */                                     \
        using TypeOf_Layout = BOOST_PP_CAT(CLASS, _parametrized);                                                      \
                                                                                                                       \
        /* Alias member types to name-derived identifyer to allow simpler definitions */                               \
        _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, cms::soa::const_cast_SoAParametersImpl, __VA_ARGS__)          \
        _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_POINTERS, ~, __VA_ARGS__)                                                 \
        _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_CONST_POINTERS, ~, __VA_ARGS__)                                           \
                                                                                                                       \
        /* Forbid copying to avoid const correctness evasion */                                                        \
        Metadata& operator=(const Metadata&) = delete;                                                                 \
        Metadata(const Metadata&) = delete;                                                                            \
                                                                                                                       \
      private:                                                                                                         \
        SOA_HOST_DEVICE SOA_INLINE Metadata(const ViewTemplateFreeParams& _soa_impl_parent)                            \
        : parent_(_soa_impl_parent) {}                                                                                 \
        const ViewTemplateFreeParams& parent_;                                                                         \
      };                                                                                                               \
                                                                                                                       \
      friend Metadata;                                                                                                 \
                                                                                                                       \
      /**                                                                                                              \
       * Helper/friend class allowing access to size from columns.                                                     \
       */                                                                                                              \
      struct Metarecords {                                                                                             \
        friend ViewTemplateFreeParams;                                                                                 \
		SOA_HOST_DEVICE SOA_INLINE\
        Metarecords(const ViewTemplateFreeParams& _soa_impl_parent) :                                                  \
                    parent_(_soa_impl_parent), _ITERATE_ON_ALL_COMMA(_STRUCT_ELEMENT_INITIALIZERS, ~, __VA_ARGS__) {}  \
        _ITERATE_ON_ALL(_ACCESSORS_STRUCT_MEMBERS, ~, __VA_ARGS__)                                                     \
        private:                                                                                                       \
          const ViewTemplateFreeParams& parent_;                                                                       \
          _ITERATE_ON_ALL(_DECLARE_STRUCT_DATA_MEMBER, ~, __VA_ARGS__)                                                 \
      };                                                                                                               \
                                                                                                                       \
      SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                           \
      SOA_HOST_DEVICE SOA_INLINE Metadata metadata() { return Metadata(*this); }                                       \
      SOA_HOST_DEVICE SOA_INLINE const Metarecords records() const { return Metarecords(*this); }                      \
      SOA_HOST_DEVICE SOA_INLINE Metarecords records() { return Metarecords(*this); }                                  \
                                                                                                                       \
      /* Trivial constuctor */                                                                                         \
      ViewTemplateFreeParams() = default;                                                                              \
                                                                                                                       \
      /* Constructor relying on user provided layout */                                                                \
      SOA_HOST_ONLY ViewTemplateFreeParams(const Metadata::TypeOf_Layout& layout)                                      \
        : base_type{layout} {}                                                                                         \
                                                                                                                       \
      /* Constructor relying on individually provided column structs */                                                \
      SOA_HOST_ONLY ViewTemplateFreeParams(                                                                            \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTOR_COLUMNS, ~, __VA_ARGS__)) {                        \
        bool readyToSet = false;                                                                                       \
        _ITERATE_ON_ALL(_INITIALIZE_VIEW_PARAMETERS_AND_SIZE, ~, __VA_ARGS__)                                          \
      }                                                                                                                \
                                                                                                                       \
      SOA_HOST_DEVICE ViewTemplateFreeParams(size_type elems,                                                          \
        _ITERATE_ON_ALL_COMMA(_DECLARE_CONSTRUCTOR_COLUMNS, ~, __VA_ARGS__)) :                                         \
        base_type{elems, _ITERATE_ON_ALL_COMMA(_INITIALIZE_COLUMNS, ~, __VA_ARGS__)} { }                               \
      /* Copiable */                                                                                                   \
      ViewTemplateFreeParams(ViewTemplateFreeParams const&) = default;                                                 \
      ViewTemplateFreeParams& operator=(ViewTemplateFreeParams const&) = default;                                      \
                                                                                                                       \
      /* Copy constructor for other parameters */                                                                      \
      template <CMS_SOA_BYTE_SIZE_TYPE OTHER_VIEW_ALIGNMENT,                                                           \
                bool OTHER_VIEW_ALIGNMENT_ENFORCEMENT,                                                                 \
                bool OTHER_RESTRICT_QUALIFY,                                                                           \
                bool OTHER_RANGE_CHECKING>                                                                             \
      ViewTemplateFreeParams(ViewTemplateFreeParams<OTHER_VIEW_ALIGNMENT, OTHER_VIEW_ALIGNMENT_ENFORCEMENT,            \
                                                    OTHER_RESTRICT_QUALIFY, OTHER_RANGE_CHECKING> const& other)        \
      : base_type{other.elements_,                                                                                     \
                  _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_OTHER_MEMBER_LIST, BOOST_PP_EMPTY(), __VA_ARGS__)                \
                } {}                                                                                                   \
      /* Copy operator for other parameters */                                                                         \
      template <CMS_SOA_BYTE_SIZE_TYPE OTHER_VIEW_ALIGNMENT,                                                           \
                bool OTHER_VIEW_ALIGNMENT_ENFORCEMENT,                                                                 \
                bool OTHER_RESTRICT_QUALIFY,                                                                           \
                bool OTHER_RANGE_CHECKING>                                                                             \
      ViewTemplateFreeParams& operator=(ViewTemplateFreeParams<OTHER_VIEW_ALIGNMENT,                                   \
        OTHER_VIEW_ALIGNMENT_ENFORCEMENT, OTHER_RESTRICT_QUALIFY, OTHER_RANGE_CHECKING> const& other)                  \
          { static_cast<base_type>(*this) = static_cast<base_type>(other); }                                           \
                                                                                                                       \
      /* Movable */                                                                                                    \
      ViewTemplateFreeParams(ViewTemplateFreeParams &&) = default;                                                     \
      ViewTemplateFreeParams& operator=(ViewTemplateFreeParams &&) = default;                                          \
                                                                                                                       \
      /* Trivial destuctor */                                                                                          \
      ~ViewTemplateFreeParams() = default;                                                                             \
                                                                                                                       \
      /* AoS-like accessor (const) */                                                                                  \
      using const_element = typename base_type::const_element;                                                         \
                                                                                                                       \
      using base_type::operator[];                                                                                     \
                                                                                                                       \
      /* AoS-like accessor (mutable) */                                                                                \
      struct element {                                                                                                 \
        SOA_HOST_DEVICE SOA_INLINE                                                                                     \
        element(size_type _soa_impl_index, /* Declare parameters */                                                    \
                _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, ~, __VA_ARGS__))                                \
            : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEM_MEMBER_INIT, _soa_impl_index, __VA_ARGS__) {}                   \
        SOA_HOST_DEVICE SOA_INLINE                                                                                     \
        element& operator=(const element& _soa_impl_other) {                                                           \
          _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_COPY, ~, __VA_ARGS__)                                            \
          return *this;                                                                                                \
        }                                                                                                              \
        SOA_HOST_DEVICE SOA_INLINE                                                                                     \
        element& operator=(const const_element& _soa_impl_other) {                                                     \
          _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_COPY, ~, __VA_ARGS__)                                            \
          return *this;                                                                                                \
        }                                                                                                              \
        /* Extra operator=() for mutable element to emulate the aggregate initialisation syntax */                     \
        SOA_HOST_DEVICE SOA_INLINE constexpr element & operator=(const typename                                        \
            BOOST_PP_CAT(CLASS, _parametrized)::Metadata::value_element _soa_impl_value) {                             \
          _ITERATE_ON_ALL(_TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT, ~, __VA_ARGS__)                                          \
          return *this;                                                                                                \
        }                                                                                                              \
                                                                                                                       \
        ENUM_IF_VALID(_ITERATE_ON_ALL(GENERATE_METHODS, ~, __VA_ARGS__))                                               \
        ENUM_IF_VALID(_ITERATE_ON_ALL(GENERATE_CONST_METHODS, ~, __VA_ARGS__))                                         \
                                                                                                                       \
        _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_MEMBER, ~, __VA_ARGS__)                                            \
      };                                                                                                               \
                                                                                                                       \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      element operator[](size_type _soa_impl_index) {                                                                  \
        if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                             \
          if (_soa_impl_index >= base_type::elements_ or _soa_impl_index < 0)                                          \
            SOA_THROW_OUT_OF_RANGE("Out of range index in ViewTemplateFreeParams" #CLASS "::operator[]",               \
              _soa_impl_index, base_type::elements_)                                                                   \
        }                                                                                                              \
        return element{_soa_impl_index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, __VA_ARGS__)};     \
      }                                                                                                                \
                                                                                                                       \
      /* inherit const accessors from ConstView */                                                                     \
                                                                                                                       \
      /* non-const accessors */                                                                                        \
      _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_ACCESSOR, ~, __VA_ARGS__)                                                      \
                                                                                                                       \
      /* dump the SoA internal structure */                                                                            \
      template <typename T>                                                                                            \
      SOA_HOST_ONLY friend void dump();                                                                                \
    };                                                                                                                 \
                                                                                                                       \
    template <bool RESTRICT_QUALIFY, bool RANGE_CHECKING>                                                              \
    using ViewTemplate = ViewTemplateFreeParams<ALIGNMENT, ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING>;   \
                                                                                                                       \
    using View = ViewTemplate<cms::soa::RestrictQualify::Default, cms::soa::RangeChecking::Default>;                   \
                                                                                                                       \
    /* Helper struct to loop over the columns without using name for non-mutable data */                               \
    struct ConstDescriptor {                                                                                           \
      ConstDescriptor() = default;                                                                                     \
                                                                                                                       \
      explicit ConstDescriptor(ConstView const& view)                                                                  \
          : buff{ _ITERATE_ON_ALL_COMMA(_ASSIGN_SPAN_TO_COLUMNS, ~, __VA_ARGS__)},                                     \
            parameterTypes{ _ITERATE_ON_ALL_COMMA(_ASSIGN_PARAMETER_TO_COLUMNS, ~, __VA_ARGS__)} {}                    \
                                                                                                                       \
      std::tuple<_ITERATE_ON_ALL_COMMA(_DECLARE_CONST_DESCRIPTOR_SPANS, ~, __VA_ARGS__)> buff;                         \
      std::tuple<_ITERATE_ON_ALL_COMMA(_DECLARE_CONST_DESCRIPTOR_PARAMETER_TYPES, ~, __VA_ARGS__)> parameterTypes;     \
      static constexpr size_type num_cols = std::tuple_size<std::tuple<                                                \
                                    _ITERATE_ON_ALL_COMMA(_DECLARE_CONST_DESCRIPTOR_SPANS, ~, __VA_ARGS__)>>::value;   \
      static constexpr std::array<cms::soa::SoAColumnType, num_cols> columnTypes = {{                                  \
        _ITERATE_ON_ALL_COMMA(_DECLARE_COLUMN_TYPE, ~, __VA_ARGS__)}};                                                 \
    };                                                                                                                 \
                                                                                                                       \
    /* Helper struct to loop over the columns without using name for mutable data */                                   \
    struct Descriptor {                                                                                                \
      Descriptor() = default;                                                                                          \
                                                                                                                       \
      explicit Descriptor(View& view)                                                                                  \
          : buff{ _ITERATE_ON_ALL_COMMA(_ASSIGN_SPAN_TO_COLUMNS, ~, __VA_ARGS__)},                                     \
            parameterTypes{ _ITERATE_ON_ALL_COMMA(_ASSIGN_PARAMETER_TO_COLUMNS, ~, __VA_ARGS__)} {}                    \
                                                                                                                       \
      std::tuple<_ITERATE_ON_ALL_COMMA(_DECLARE_DESCRIPTOR_SPANS, ~, __VA_ARGS__)> buff;                               \
      std::tuple<_ITERATE_ON_ALL_COMMA(_DECLARE_DESCRIPTOR_PARAMETER_TYPES, ~, __VA_ARGS__)> parameterTypes;           \
      static constexpr size_type num_cols = std::tuple_size<std::tuple<                                                \
                                    _ITERATE_ON_ALL_COMMA(_DECLARE_DESCRIPTOR_SPANS, ~, __VA_ARGS__)>>::value;         \
      static constexpr std::array<cms::soa::SoAColumnType, num_cols> columnTypes = {{                                  \
        _ITERATE_ON_ALL_COMMA(_DECLARE_COLUMN_TYPE, ~, __VA_ARGS__)}};                                                 \
    };                                                                                                                 \
                                                                                                                       \
    /* Trivial constuctor */                                                                                           \
    CLASS()                                                                                                            \
        : mem_(nullptr),                                                                                               \
          elements_(0),                                                                                                \
          byteSize_(0),                                                                                                \
          _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_TRIVIAL_CONSTRUCTION, ~, __VA_ARGS__) {}                               \
                                                                                                                       \
    /* Constructor relying on user provided storage (implementation shared with ROOT streamer) */                      \
    SOA_HOST_ONLY CLASS(std::byte* mem, size_type elements) : mem_(mem), elements_(elements), byteSize_(0) {           \
      organizeColumnsFromBuffer();                                                                                     \
    }                                                                                                                  \
                                                                                                                       \
    /* Explicit copy constructor and assignment operator */                                                            \
    SOA_HOST_ONLY CLASS(CLASS const& _soa_impl_other)                                                                  \
        : mem_(_soa_impl_other.mem_),                                                                                  \
          elements_(_soa_impl_other.elements_),                                                                        \
          byteSize_(_soa_impl_other.byteSize_),                                                                        \
          _ITERATE_ON_ALL_COMMA(_DECLARE_MEMBER_COPY_CONSTRUCTION, ~, __VA_ARGS__) {}                                  \
                                                                                                                       \
    SOA_HOST_ONLY CLASS& operator=(CLASS const& _soa_impl_other) {                                                     \
        mem_ = _soa_impl_other.mem_;                                                                                   \
        elements_ = _soa_impl_other.elements_;                                                                         \
        byteSize_ = _soa_impl_other.byteSize_;                                                                         \
        _ITERATE_ON_ALL(_DECLARE_MEMBER_ASSIGNMENT, ~, __VA_ARGS__)                                                    \
        return *this;                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper to implement View as derived from ConstView in SoABlocks implementation */                               \
    template <bool RESTRICT_QUALIFY, bool RANGE_CHECKING>                                                              \
    SOA_HOST_DEVICE SOA_INLINE static ViewTemplate<RESTRICT_QUALIFY, RANGE_CHECKING> const_cast_View(                  \
      ConstViewTemplate<RESTRICT_QUALIFY, RANGE_CHECKING> const& view)  {                                              \
      return ViewTemplate<RESTRICT_QUALIFY, RANGE_CHECKING>{                                                           \
        view.metadata().size(), _ITERATE_ON_ALL_COMMA(_DECLARE_CONST_CAST_COLUMNS, ~, __VA_ARGS__)};                   \
    }                                                                                                                  \
                                                                                                                       \
    /*                                                                                                                 \
     * Method for copying the data from a generic View to a memory blob.                                               \
     * Host-only data can be handled by this method.                                                                   \
     */                                                                                                                \
    SOA_HOST_ONLY void deepCopy(ConstView const& view) {                                                               \
      if (elements_ < view.metadata().size())                                                                          \
        throw std::runtime_error(                                                                                      \
            "In "#CLASS"::deepCopy method: number of elements mismatch ");                                             \
      _ITERATE_ON_ALL(_COPY_VIEW_COLUMNS, ~, __VA_ARGS__)                                                              \
    }                                                                                                                  \
                                                                                                                       \
    /* ROOT read streamer */                                                                                           \
    template <typename T>                                                                                              \
    void ROOTReadStreamer(T & onfile) {                                                                                \
      _ITERATE_ON_ALL(_STREAMER_READ_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* ROOT allocation cleanup */                                                                                      \
    void ROOTStreamerCleaner() {                                                                                       \
      /* This function should only be called from the PortableCollection ROOT streamer */                              \
      _ITERATE_ON_ALL(_ROOT_FREE_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                 \
    }                                                                                                                  \
                                                                                                                       \
    /* Dump the SoA internal structure */                                                                              \
    template <typename T>                                                                                              \
    SOA_HOST_ONLY friend void dump();                                                                                  \
                                                                                                                       \
  private:                                                                                                             \
    /* Helper method for the user provided storage constructor and ROOT streamer */                                    \
    void organizeColumnsFromBuffer() {                                                                                 \
      if constexpr (alignmentEnforcement == cms::soa::AlignmentEnforcement::enforced)                                  \
        if (reinterpret_cast<intptr_t>(mem_) % alignment)                                                              \
          throw std::runtime_error("In " #CLASS "::" #CLASS ": misaligned buffer");                                    \
      auto _soa_impl_curMem = mem_;                                                                                    \
      _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                    \
      /* Sanity check: we should have reached the computed size, only on host code */                                  \
      byteSize_ = computeDataSize(elements_);                                                                          \
      if (mem_ + byteSize_ != _soa_impl_curMem)                                                                        \
        throw std::runtime_error("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                \
    }                                                                                                                  \
                                                                                                                       \
    /* Helper function to compute the total number of methods */                                                       \
    static constexpr std::pair<size_type, size_type> computeMethodsNumber() {                                          \
      size_type _soa_methods_count = 0;                                                                                \
      size_type _soa_const_methods_count = 0;                                                                          \
                                                                                                                       \
      _ITERATE_ON_ALL(_COUNT_SOA_METHODS, _soa_methods_count, __VA_ARGS__)                                             \
      _ITERATE_ON_ALL(_COUNT_SOA_CONST_METHODS, _soa_const_methods_count, __VA_ARGS__)                                 \
                                                                                                                       \
      return {_soa_methods_count, _soa_const_methods_count};                                                           \
    }                                                                                                                  \
                                                                                                                       \
    /* compile-time error launched if more than one macro for methods is declared */                                   \
    static_assert(computeMethodsNumber().first <= 1,                                                                   \
                  "There can be at most one SOA_ELEMENT_METHODS macro."                                                \
                  "Please declare all your methods inside the same macro.");                                           \
                                                                                                                       \
    static_assert(computeMethodsNumber().second <= 1,                                                                  \
                  "There can be at most one SOA_CONST_ELEMENT_METHODS macro."                                          \
                  "Please declare all your methods inside the same macro.");                                           \
                                                                                                                       \
    /* Data members */                                                                                                 \
    std::byte* mem_ EDM_REFLEX_TRANSIENT;                                                                              \
    size_type elements_;                                                                                               \
    size_type const scalar_ = 1;                                                                                       \
    byte_size_type byteSize_ EDM_REFLEX_TRANSIENT;                                                                     \
    /* TODO: The layout will contain SoAParametersImpl as members for the columns, which will allow the use of   	   \
    * more template helper functions. */																			   \
    _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                          \
    /* Making the code conditional is problematic in macros as the commas will interfere with parameter lisings     */ \
    /* So instead we make the code unconditional with paceholder names which are protected by a private protection. */ \
    /* This will be handled later as we handle the integration of the view as a subclass of the layout.             */ \
  };
// clang-format on

#endif  // DataFormats_SoATemplate_interface_SoALayout_h
