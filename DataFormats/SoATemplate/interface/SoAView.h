#ifndef DataFormats_SoATemplate_interface_SoAView_h
#define DataFormats_SoATemplate_interface_SoAView_h

/*
 * Structure-of-Arrays templates allowing access to a selection of scalars and columns from one
 * or multiple SoA layouts or views.
 * This template generator will allow handling subsets of columns from one or multiple SoA views or layouts.
 */

#include "SoACommon.h"

#define SOA_VIEW_LAYOUT(TYPE, NAME) (TYPE, NAME)

#define SOA_VIEW_LAYOUT_LIST(...) __VA_ARGS__

#define SOA_VIEW_VALUE(LAYOUT_NAME, LAYOUT_MEMBER) (LAYOUT_NAME, LAYOUT_MEMBER, LAYOUT_MEMBER)

#define SOA_VIEW_VALUE_RENAME(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME) (LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)

#define SOA_VIEW_VALUE_LIST(...) __VA_ARGS__

/*
 * A macro defining a SoA view (collection of columns from multiple layouts or views.)
 *
 * Usage:
 * GENERATE_SOA_VIEW(PixelXYConstView, PixelXYView,
 *   SOA_VIEW_LAYOUT_LIST(
 *     SOA_VIEW_LAYOUT(PixelDigis,         pixelDigis),
 *     SOA_VIEW_LAYOUT(PixelRecHitsLayout, pixelsRecHit)
 *   ),
 *   SOA_VIEW_VALUE_LIST(
 *     SOA_VIEW_VALUE_RENAME(pixelDigis,   x,   digisX),
 *     SOA_VIEW_VALUE_RENAME(pixelDigis,   y,   digisY),
 *     SOA_VIEW_VALUE_RENAME(pixelsRecHit, x, recHitsX),
 *     SOA_VIEW_VALUE_RENAME(pixelsRecHit, y, recHitsY)
 *   )
 * );
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

}  // namespace cms::soa

/*
 * Members definitions macros for views
 */

/**
 * Layout templates parametrization
 */
#define _DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE_IMPL(TYPE, NAME)                            \
  (using BOOST_PP_CAT(TYPE, _default) = BOOST_PP_CAT(TYPE, _StagedTemplates) < VIEW_ALIGNMENT, \
   VIEW_ALIGNMENT_ENFORCEMENT > ;)

#define _DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE_IMPL TYPE_NAME)

/**
 * Layout types aliasing for referencing by name
 */
#define _DECLARE_VIEW_LAYOUT_TYPE_ALIAS_IMPL(TYPE, NAME) using BOOST_PP_CAT(TypeOf_, NAME) = TYPE;

#define _DECLARE_VIEW_LAYOUT_TYPE_ALIAS(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_LAYOUT_TYPE_ALIAS_IMPL TYPE_NAME)

/**
 * Member types aliasing for referencing by name
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, CAST)                             \
  using BOOST_PP_CAT(TypeOf_, LOCAL_NAME) =                                                                            \
      typename BOOST_PP_CAT(TypeOf_, LAYOUT_NAME)::Metadata::BOOST_PP_CAT(TypeOf_, LAYOUT_MEMBER);                     \
  using BOOST_PP_CAT(ParametersTypeOf_, LOCAL_NAME) =                                                                  \
      typename BOOST_PP_CAT(TypeOf_, LAYOUT_NAME)::Metadata::BOOST_PP_CAT(ParametersTypeOf_, LAYOUT_MEMBER);           \
  constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME) =                                   \
      BOOST_PP_CAT(TypeOf_, LAYOUT_NAME)::Metadata::BOOST_PP_CAT(ColumnTypeOf_, LAYOUT_MEMBER);                        \
  using BOOST_PP_CAT(ConstAccessorOf_, LOCAL_NAME) =                                                                   \
    typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                            \
        template ColumnType<BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                             \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                          \
                template RestrictQualifier<restrictQualify> ;                                                          \
  using BOOST_PP_CAT(MutableAccessorOf_, LOCAL_NAME) =                                                                 \
    typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                            \
        template ColumnType<BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                             \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify> ;                                                          \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
  const auto BOOST_PP_CAT(parametersOf_, LOCAL_NAME)() const {                                                         \
    return CAST(parent_.BOOST_PP_CAT(LOCAL_NAME, Parameters_));                                                        \
  };
// clang-format on

// DATA should be a function used to convert
//   parent_.LOCAL_NAME ## Parameters_
// to
//   ParametersTypeOf_ ## LOCAL_NAME                (for a View)
// or
//   ParametersTypeOf_ ## LOCAL_NAME :: ConstType   (for a ConstView)
// or empty, if no conversion is necessary.
#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Member type pointers for referencing by name
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_POINTERS_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                                     \
  SOA_HOST_DEVICE SOA_INLINE auto* BOOST_PP_CAT(addressOf_, LOCAL_NAME)() {                                            \
    return BOOST_PP_CAT(parametersOf_, LOCAL_NAME)().addr_;                                                            \
  };
// clang-format on

#define _DECLARE_VIEW_MEMBER_POINTERS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_POINTERS_IMPL LAYOUT_MEMBER_NAME)

/**
 * Member type const pointers for referencing by name
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_CONST_POINTERS_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                               \
  SOA_HOST_DEVICE SOA_INLINE auto const* BOOST_PP_CAT(addressOf_, LOCAL_NAME)() const {                                \
    return BOOST_PP_CAT(parametersOf_, LOCAL_NAME)().addr_;                                                            \
  };
// clang-format on

#define _DECLARE_VIEW_MEMBER_CONST_POINTERS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_CONST_POINTERS_IMPL LAYOUT_MEMBER_NAME)

/**
 * Generator of parameters (layouts/views) for constructor by layouts/views.
 */
#define _DECLARE_VIEW_CONSTRUCTION_PARAMETERS_IMPL(LAYOUT_TYPE, LAYOUT_NAME, DATA) (DATA LAYOUT_TYPE & LAYOUT_NAME)

#define _DECLARE_VIEW_CONSTRUCTION_PARAMETERS(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))

/**
 * Generator of parameters for constructor by column.
 */
#define _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  (DATA typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, LOCAL_NAME)::TupleOrPointerType LOCAL_NAME)

#define _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(                                                                  \
      _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL(LAYOUT, MEMBER, NAME)                                                   \
  (BOOST_PP_CAT(NAME, Parameters_)([&]() -> auto {                                                                     \
    auto params = LAYOUT.metadata().BOOST_PP_CAT(parametersOf_, MEMBER)();                                             \
    if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                              \
      if (reinterpret_cast<intptr_t>(params.addr_) % alignment)                                                        \
        throw std::runtime_error("In constructor by layout: misaligned column: " #NAME);                               \
    return params;                                                                                                     \
  }()))
// clang-format on

#define _DECLARE_VIEW_MEMBER_INITIALIZERS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL LAYOUT_MEMBER_NAME)

/**
 * Generator of size computation for constructor.
 * This is the per-layout part of the lambda checking they all have the same size.
 */
// clang-format off
#define _UPDATE_SIZE_OF_VIEW_IMPL(LAYOUT_TYPE, LAYOUT_NAME)                                                            \
  if (set) {                                                                                                           \
    if (ret != LAYOUT_NAME.metadata().size())                                                                          \
      throw std::runtime_error("In constructor by layout: different sizes from layouts.");                             \
  } else {                                                                                                             \
    ret = LAYOUT_NAME.metadata().size();                                                                               \
    set = true;                                                                                                        \
  }
// clang-format on

#define _UPDATE_SIZE_OF_VIEW(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_UPDATE_SIZE_OF_VIEW_IMPL TYPE_NAME)

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN_IMPL(LAYOUT, MEMBER, NAME)                                          \
  (                                                                                                                    \
    BOOST_PP_CAT(NAME, Parameters_)([&]() -> auto {                                                                    \
      if constexpr (alignmentEnforcement == AlignmentEnforcement::enforced)                                            \
        if (Metadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::checkAlignment(NAME, alignment))                         \
          throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                             \
      return NAME;                                                                                                     \
    }())                                                                                                               \
  )
// clang-format on

#define _DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN_IMPL LAYOUT_MEMBER_NAME)

/**
 * Generator of layout list.
 */
#define _DECLARE_LAYOUT_LIST_IMPL(LAYOUT, NAME) (NAME)

#define _DECLARE_LAYOUT_LIST(R, DATA, LAYOUT_MEMBER_NAME) BOOST_PP_EXPAND(_DECLARE_LAYOUT_LIST_IMPL LAYOUT_MEMBER_NAME)

/**
 * Generator of view member list.
 */
#define _DECLARE_VIEW_MEMBER_LIST_IMPL(LAYOUT, MEMBER, NAME) (NAME)

#define _DECLARE_VIEW_MEMBER_LIST(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_LIST_IMPL LAYOUT_MEMBER_NAME)

/**
 * Generator of member initializer for copy constructor.
 */
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_FROM_OTHER_IMPL(LAYOUT, MEMBER, LOCAL_NAME, DATA) \
  (BOOST_PP_CAT(MEMBER, Parameters_){DATA.BOOST_PP_CAT(MEMBER, Parameters_)})

#define _DECLARE_VIEW_MEMBER_INITIALIZERS_FROM_OTHER(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_FROM_OTHER_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Generator of member assignment for assignment operator.
 */
#define _DECLARE_VIEW_MEMBER_ASSIGNMENT_FROM_OTHER_IMPL(LAYOUT, MEMBER, LOCAL_NAME, DATA) \
  BOOST_PP_CAT(MEMBER, Parameters_) = DATA.BOOST_PP_CAT(MEMBER, Parameters_);

#define _DECLARE_VIEW_MEMBER_ASSIGNMENT_FROM_OTHER(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_ASSIGNMENT_FROM_OTHER_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Generator of element members initializer.
 */
#define _DECLARE_VIEW_ELEM_MEMBER_INIT_IMPL(LAYOUT, MEMBER, LOCAL_NAME, DATA) (LOCAL_NAME(DATA, LOCAL_NAME))

#define _DECLARE_VIEW_ELEM_MEMBER_INIT(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEM_MEMBER_INIT_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Helper macro extracting the data type from metadata of a layout or view
 */
#define _COLUMN_TYPE(LAYOUT_NAME, LAYOUT_MEMBER) \
  typename std::remove_pointer<decltype(BOOST_PP_CAT(LAYOUT_NAME, Type)()::LAYOUT_MEMBER())>::type

/**
 * Generator of parameters for (non-const) element subclass (expanded comma separated).
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  (DATA typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, LOCAL_NAME) LOCAL_NAME)

#define _DECLARE_VIEW_ELEMENT_VALUE_ARG(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA)

/**
 * Generator of parameters for (const) element subclass (expanded comma separated).
 */
#define _DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  (DATA typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, LOCAL_NAME)::ConstType LOCAL_NAME)

#define _DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA)

/**
 * Generator of member initialization for constructor of element subclass
 */
#define _DECLARE_VIEW_CONST_ELEM_MEMBER_INIT_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  (BOOST_PP_CAT(LOCAL_NAME, _)(DATA, LOCAL_NAME))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_VIEW_CONST_ELEM_MEMBER_INIT(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Declaration of the members accessors of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                              \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
      const typename SoAConstValueWithConf<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME),                          \
      const typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::RefToConst                                          \
      LOCAL_NAME() const {                                                                                             \
    return BOOST_PP_CAT(LOCAL_NAME, _)();                                                                              \
  }
// clang-format on

#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL LAYOUT_MEMBER_NAME

/**
 * Declaration of the private members of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                          \
  const cms::soa::ConstValueTraits<SoAConstValueWithConf<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME),            \
                                                         typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>,        \
                                   BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>                                  \
      BOOST_PP_CAT(LOCAL_NAME, _);
// clang-format on

#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL LAYOUT_MEMBER_NAME

/**
 * Generator of the member-by-member copy operator of the element subclass.
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                 \
  if constexpr (Metadata::BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME) != cms::soa::SoAColumnType::scalar) \
    LOCAL_NAME() = other.LOCAL_NAME();

#define _DECLARE_VIEW_ELEMENT_VALUE_COPY(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL LAYOUT_MEMBER_NAME)

/**
 * Declaration of the private members of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                                \
  SoAValueWithConf<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME),                                                  \
                   typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>                                               \
      LOCAL_NAME;
// clang-format on

#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL LAYOUT_MEMBER_NAME

/**
 * Parameters passed to const element subclass constructor in operator[]
 */
#define _DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME) \
  (BOOST_PP_CAT(LOCAL_NAME, Parameters_))

#define _DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL_IMPL LAYOUT_MEMBER_NAME)

/**
 * Parameters passed to element subclass constructor in operator[]
 *
 * The use of const_cast (inside const_cast_SoAParametersImpl) is safe because the constructor of a View binds only to
 * non-const arguments.
 */
#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME) \
  (const_cast_SoAParametersImpl(base_type::BOOST_PP_CAT(LOCAL_NAME, Parameters_)))

#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL LAYOUT_MEMBER_NAME)

/**
 * Direct access to column pointer and indexed access
 */
// clang-format off
#define _DECLARE_VIEW_SOA_ACCESSOR_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                                        \
  /* Column or scalar */                                                                                               \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                              \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::NoParamReturnType                                        \
  LOCAL_NAME() {                                                                                                       \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                     \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(const_cast_SoAParametersImpl(                              \
                    base_type:: BOOST_PP_CAT(LOCAL_NAME, Parameters_)))();                                             \
  }                                                                                                                    \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                              \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                        \
                 template RestrictQualifier<restrictQualify>::ParamReturnType                                          \
  LOCAL_NAME(size_type index) {                                                                                        \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                                 \
      if (index >= base_type::elements_)                                                                               \
        SOA_THROW_OUT_OF_RANGE("Out of range index in mutable " #LOCAL_NAME "(size_type index)")                       \
    }                                                                                                                  \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                     \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::mutableAccess>::template Alignment<conditionalAlignment>::                        \
                template RestrictQualifier<restrictQualify>(const_cast_SoAParametersImpl(                              \
                    base_type:: BOOST_PP_CAT(LOCAL_NAME, Parameters_)))(index);                                        \
  }
// clang-format on

#define _DECLARE_VIEW_SOA_ACCESSOR(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_ACCESSOR_IMPL LAYOUT_MEMBER_NAME)

/**
 * Direct access to column pointer (const) and indexed access.
 */
// clang-format off
#define _DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                                  \
  /* Column or scalar */                                                                                               \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                              \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                          \
                template RestrictQualifier<restrictQualify>::NoParamReturnType                                         \
  LOCAL_NAME() const {                                                                                                 \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                     \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                          \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(LOCAL_NAME, Parameters_))();                  \
  }                                                                                                                    \
  SOA_HOST_DEVICE SOA_INLINE                                                                                           \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                              \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                          \
                template RestrictQualifier<restrictQualify>::ParamReturnType                                           \
  LOCAL_NAME(size_type index) const {                                                                                  \
    if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                                 \
      if (index >= elements_)                                                                                          \
        SOA_THROW_OUT_OF_RANGE("Out of range index in const " #LOCAL_NAME "(size_type index)")                         \
    }                                                                                                                  \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(Metadata::TypeOf_, LOCAL_NAME)>::                     \
        template ColumnType<BOOST_PP_CAT(Metadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType<                   \
            cms::soa::SoAAccessType::constAccess>::template Alignment<conditionalAlignment>::                          \
                template RestrictQualifier<restrictQualify>(BOOST_PP_CAT(LOCAL_NAME, Parameters_))(index);             \
  }
// clang-format on

#define _DECLARE_VIEW_SOA_CONST_ACCESSOR(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL LAYOUT_MEMBER_NAME)

/**
 * SoA class member declaration (column pointers and parameters).
 */
#define _DECLARE_VIEW_SOA_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, LOCAL_NAME) BOOST_PP_CAT(LOCAL_NAME, Parameters_);

#define _DECLARE_VIEW_SOA_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_MEMBER_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Const SoA class member declaration (column pointers and parameters).
 */
#define _DECLARE_CONST_VIEW_SOA_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  typename BOOST_PP_CAT(Metadata::ParametersTypeOf_, LOCAL_NAME)::ConstType BOOST_PP_CAT(LOCAL_NAME, Parameters_);

#define _DECLARE_CONST_VIEW_SOA_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_SOA_MEMBER_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Assign the value of the view from the values in the value_element.
 */

// clang-format off
#define _TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT_IMPL(VALUE_TYPE, CPP_TYPE, NAME)                                            \
  _SWITCH_ON_TYPE(VALUE_TYPE,                                                                                          \
      /* Scalar (empty) */                                                                                             \
      ,                                                                                                                \
      /* Column */                                                                                                     \
      NAME() = value.NAME;                                                                                             \
      ,                                                                                                                \
      /* Eigen column */                                                                                               \
      NAME() = value.NAME;                                                                                             \
)
// clang-format on

#define _TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT(R, DATA, TYPE_NAME) _TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT_IMPL TYPE_NAME

/* ---- MUTABLE VIEW ------------------------------------------------------------------------------------------------ */
// clang-format off
#define _GENERATE_SOA_VIEW_PART_0(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                          \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                              \
            bool VIEW_ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,                                 \
            bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,                                                \
            bool RANGE_CHECKING = cms::soa::RangeChecking::disabled>                                                   \
  struct VIEW : public CONST_VIEW<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING> {      \
    /* Declare the parametrized layouts as the default */                                                              \
    /*BOOST_PP_SEQ_CAT(_ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE, ~, LAYOUTS_LIST))   */              \
    /* these could be moved to an external type trait to free up the symbol names */                                   \
    using self_type = VIEW;                                                                                            \
    using base_type = CONST_VIEW<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING>;
// clang-format on

// clang-format off
#define _GENERATE_SOA_VIEW_PART_0_NO_DEFAULTS(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                              \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                     \
            bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                           \
            bool RESTRICT_QUALIFY,                                                                                     \
            bool RANGE_CHECKING>                                                                                       \
  struct VIEW : public CONST_VIEW<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING> {      \
    /* Declare the parametrized layouts as the default */                                                              \
    /*BOOST_PP_SEQ_CAT(_ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE, ~, LAYOUTS_LIST))   */              \
    /* these could be moved to an external type trait to free up the symbol names */                                   \
    using self_type = VIEW;                                                                                            \
    using base_type = CONST_VIEW<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT, RESTRICT_QUALIFY, RANGE_CHECKING>;
// clang-format on

/**
 * Split of the const view definition where the parametrized template alias for the layout is defined for layout trivial view.
 */

// clang-format off
#define _GENERATE_SOA_VIEW_PART_1(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                          \
    using size_type = cms::soa::size_type;                                                                             \
    using byte_size_type = cms::soa::byte_size_type;                                                                   \
    using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                       \
                                                                                                                       \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                            \
     * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid      \
     * up to compute capability 8.X.                                                                                   \
     */                                                                                                                \
    constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;                           \
    constexpr static byte_size_type alignment = VIEW_ALIGNMENT;                                                        \
    constexpr static bool alignmentEnforcement = VIEW_ALIGNMENT_ENFORCEMENT;                                           \
    constexpr static byte_size_type conditionalAlignment =                                                             \
        alignmentEnforcement == AlignmentEnforcement::enforced ? alignment : 0;                                        \
    constexpr static bool restrictQualify = RESTRICT_QUALIFY;                                                          \
    constexpr static bool rangeChecking = RANGE_CHECKING;                                                              \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                          \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;                \
                                                                                                                       \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;      \
                                                                                                                       \
    /**                                                                                                                \
     * Helper/friend class allowing SoA introspection.                                                                 \
     */                                                                                                                \
    struct Metadata {                                                                                                  \
      friend VIEW;                                                                                                     \
      SOA_HOST_DEVICE SOA_INLINE size_type size() const { return parent_.elements_; }                                  \
      /* Alias layout or view types to name-derived identifyer to allow simpler definitions */                         \
      _ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_TYPE_ALIAS, ~, LAYOUTS_LIST)                                                \
                                                                                                                       \
      /* Alias member types to name-derived identifyer to allow simpler definitions */                                 \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, const_cast_SoAParametersImpl, VALUE_LIST)                       \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_POINTERS, ~, VALUE_LIST)                                                    \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_CONST_POINTERS, ~, VALUE_LIST)                                              \
                                                                                                                       \
      /* Forbid copying to avoid const correctness evasion */                                                          \
      Metadata& operator=(const Metadata&) = delete;                                                                   \
      Metadata(const Metadata&) = delete;                                                                              \
                                                                                                                       \
    private:                                                                                                           \
      SOA_HOST_DEVICE SOA_INLINE Metadata(const VIEW& parent) : parent_(parent) {}                                     \
      const VIEW& parent_;                                                                                             \
    };                                                                                                                 \
                                                                                                                       \
    friend Metadata;                                                                                                   \
    SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                             \
    SOA_HOST_DEVICE SOA_INLINE Metadata metadata() { return Metadata(*this); }                                         \
                                                                                                                       \
    /* Trivial constuctor */                                                                                           \
    VIEW() = default;                                                                                                  \
                                                                                                                       \
    /* Constructor relying on user provided layouts or views */                                                        \
    SOA_HOST_ONLY VIEW(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS, BOOST_PP_EMPTY(), LAYOUTS_LIST))   \
        : base_type{_ITERATE_ON_ALL_COMMA(_DECLARE_LAYOUT_LIST, BOOST_PP_EMPTY(), LAYOUTS_LIST)} {}                    \
                                                                                                                       \
    /* Constructor relying on individually provided column addresses */                                                \
    SOA_HOST_ONLY VIEW(size_type elements,                                                                             \
                        _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS,                          \
                                              BOOST_PP_EMPTY(),                                                        \
                                              VALUE_LIST))                                                             \
        : base_type{elements, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_LIST, BOOST_PP_EMPTY(), VALUE_LIST)} {}       \
                                                                                                                       \
    /* Copiable */                                                                                                     \
    VIEW(VIEW const&) = default;                                                                                       \
    VIEW& operator=(VIEW const&) = default;                                                                            \
                                                                                                                       \
    /* Movable */                                                                                                      \
    VIEW(VIEW &&) = default;                                                                                           \
    VIEW& operator=(VIEW &&) = default;                                                                                \
                                                                                                                       \
    /* Trivial destuctor */                                                                                            \
    ~VIEW() = default;                                                                                                 \
                                                                                                                       \
    /* AoS-like accessor (const) */                                                                                    \
    using const_element = typename base_type::const_element;                                                           \
                                                                                                                       \
    using base_type::operator[];                                                                                       \
                                                                                                                       \
    /* AoS-like accessor (mutable) */                                                                                  \
    struct element {                                                                                                   \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      element(size_type index, /* Declare parameters */                                                                \
              _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, BOOST_PP_EMPTY(), VALUE_LIST))                    \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      element& operator=(const element& other) {                                                                       \
        _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_COPY, ~, VALUE_LIST)                                               \
        return *this;                                                                                                  \
      }
// clang-format on

// clang-format off
#define _GENERATE_SOA_VIEW_PART_2(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                          \
      _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                               \
    };                                                                                                                 \
                                                                                                                       \
    SOA_HOST_DEVICE SOA_INLINE                                                                                         \
    element operator[](size_type index) {                                                                              \
      if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
        if (index >= base_type::elements_)                                                                             \
          SOA_THROW_OUT_OF_RANGE("Out of range index in " #VIEW "::operator[]")                                        \
      }                                                                                                                \
      return element{index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST)};                  \
    }                                                                                                                  \
                                                                                                                       \
    /* inherit const accessors from ConstView */                                                                       \
                                                                                                                       \
    /* non-const accessors */                                                                                          \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_ACCESSOR, ~, VALUE_LIST)                                                         \
                                                                                                                       \
    /* dump the SoA internal structure */                                                                              \
    template <typename T>                                                                                              \
    SOA_HOST_ONLY friend void dump();                                                                                  \
  };
// clang-format on

/* ---- CONST VIEW -------------------------------------------------------------------------------------------------- */
// clang-format off
#define _GENERATE_SOA_CONST_VIEW_PART_0(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                    \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                              \
            bool VIEW_ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::relaxed,                                 \
            bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::enabled,                                                \
            bool RANGE_CHECKING = cms::soa::RangeChecking::disabled>                                                   \
  struct CONST_VIEW {                                                                                                  \
    /* these could be moved to an external type trait to free up the symbol names */                                   \
    using self_type = CONST_VIEW;
// clang-format on

// clang-format off
#define _GENERATE_SOA_CONST_VIEW_PART_0_NO_DEFAULTS(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                        \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                     \
            bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                           \
            bool RESTRICT_QUALIFY,                                                                                     \
            bool RANGE_CHECKING>                                                                                       \
  struct CONST_VIEW {                                                                                                  \
    /* these could be moved to an external type trait to free up the symbol names */                                   \
    using self_type = CONST_VIEW;
// clang-format on

/**
 * Split of the const view definition where the parametrized template alias for the layout is defined for layout trivial view.
 */

// clang-format off
#define _GENERATE_SOA_CONST_VIEW_PART_1(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                    \
    using size_type = cms::soa::size_type;                                                                             \
    using byte_size_type = cms::soa::byte_size_type;                                                                   \
    using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                       \
                                                                                                                       \
    template <CMS_SOA_BYTE_SIZE_TYPE, bool, bool, bool>                                                                \
    friend struct VIEW;                                                                                                \
                                                                                                                       \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                            \
     * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid      \
     * up to compute capability 8.X.                                                                                   \
     */                                                                                                                \
    constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;                           \
    constexpr static byte_size_type alignment = VIEW_ALIGNMENT;                                                        \
    constexpr static bool alignmentEnforcement = VIEW_ALIGNMENT_ENFORCEMENT;                                           \
    constexpr static byte_size_type conditionalAlignment =                                                             \
        alignmentEnforcement == AlignmentEnforcement::enforced ? alignment : 0;                                        \
    constexpr static bool restrictQualify = RESTRICT_QUALIFY;                                                          \
    constexpr static bool rangeChecking = RANGE_CHECKING;                                                              \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                          \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;                \
                                                                                                                       \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                            \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;      \
                                                                                                                       \
    /**                                                                                                                \
     * Helper/friend class allowing SoA introspection.                                                                 \
     */                                                                                                                \
    struct Metadata {                                                                                                  \
      friend CONST_VIEW;                                                                                               \
      SOA_HOST_DEVICE SOA_INLINE size_type size() const { return parent_.elements_; }                                  \
      /* Alias layout or view types to name-derived identifyer to allow simpler definitions */                         \
      _ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_TYPE_ALIAS, ~, LAYOUTS_LIST)                                                \
                                                                                                                       \
      /* Alias member types to name-derived identifyer to allow simpler definitions */                                 \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, BOOST_PP_EMPTY(), VALUE_LIST)                                   \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_CONST_POINTERS, ~, VALUE_LIST)                                              \
                                                                                                                       \
      /* Forbid copying to avoid const correctness evasion */                                                          \
      Metadata& operator=(const Metadata&) = delete;                                                                   \
      Metadata(const Metadata&) = delete;                                                                              \
                                                                                                                       \
    private:                                                                                                           \
      SOA_HOST_DEVICE SOA_INLINE Metadata(const CONST_VIEW& parent) : parent_(parent) {}                               \
      const CONST_VIEW& parent_;                                                                                       \
    };                                                                                                                 \
                                                                                                                       \
    friend Metadata;                                                                                                   \
    SOA_HOST_DEVICE SOA_INLINE const Metadata metadata() const { return Metadata(*this); }                             \
                                                                                                                       \
    /* Trivial constuctor */                                                                                           \
    CONST_VIEW() = default;                                                                                            \
                                                                                                                       \
    /* Constructor relying on user provided layouts or views */                                                        \
    SOA_HOST_ONLY CONST_VIEW(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS, const, LAYOUTS_LIST))        \
        : elements_([&]() -> size_type {                                                                               \
            bool set = false;                                                                                          \
            size_type ret = 0;                                                                                         \
            _ITERATE_ON_ALL(_UPDATE_SIZE_OF_VIEW, BOOST_PP_EMPTY(), LAYOUTS_LIST)                                      \
            return ret;                                                                                                \
          }()),                                                                                                        \
          _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS, ~, VALUE_LIST) {}                                   \
                                                                                                                       \
    /* Constructor relying on individually provided column addresses */                                                \
    SOA_HOST_ONLY CONST_VIEW(size_type elements,                                                                       \
                        _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS, const, VALUE_LIST))      \
        : elements_(elements), _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN, ~, VALUE_LIST) {}     \
                                                                                                                       \
    /* Copiable */                                                                                                     \
    CONST_VIEW(CONST_VIEW const&) = default;                                                                           \
    CONST_VIEW& operator=(CONST_VIEW const&) = default;                                                                \
                                                                                                                       \
    /* Movable */                                                                                                      \
    CONST_VIEW(CONST_VIEW &&) = default;                                                                               \
    CONST_VIEW& operator=(CONST_VIEW &&) = default;                                                                    \
                                                                                                                       \
    /* Trivial destuctor */                                                                                            \
    ~CONST_VIEW() = default;                                                                                           \
                                                                                                                       \
    /* AoS-like accessor (const) */                                                                                    \
    struct const_element {                                                                                             \
      SOA_HOST_DEVICE SOA_INLINE                                                                                       \
      const_element(size_type index, /* Declare parameters */                                                          \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG, const, VALUE_LIST))                   \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                          \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_ACCESSOR, ~, VALUE_LIST)                                             \
                                                                                                                       \
    private:                                                                                                           \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                         \
    };                                                                                                                 \
                                                                                                                       \
    SOA_HOST_DEVICE SOA_INLINE                                                                                         \
    const_element operator[](size_type index) const {                                                                  \
      if constexpr (rangeChecking == cms::soa::RangeChecking::enabled) {                                               \
        if (index >= elements_)                                                                                        \
          SOA_THROW_OUT_OF_RANGE("Out of range index in " #CONST_VIEW "::operator[]")                                  \
      }                                                                                                                \
      return const_element{index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEMENT_CONSTR_CALL, ~, VALUE_LIST)};      \
    }                                                                                                                  \
                                                                                                                       \
    /* const accessors */                                                                                              \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_CONST_ACCESSOR, ~, VALUE_LIST)                                                   \
                                                                                                                       \
    /* dump the SoA internal structure */                                                                              \
    template <typename T>                                                                                              \
    SOA_HOST_ONLY friend void dump();                                                                                  \
                                                                                                                       \
  private:                                                                                                             \
    size_type elements_ = 0;                                                                                           \
    _ITERATE_ON_ALL(_DECLARE_CONST_VIEW_SOA_MEMBER, const, VALUE_LIST)                                                 \
};
// clang-format on

// clang-format off
// MAJOR caveat: in order to propagate the LAYOUTS_LIST and VALUE_LIST
#define _GENERATE_SOA_CONST_VIEW(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                           \
   _GENERATE_SOA_CONST_VIEW_PART_0(CONST_VIEW, VIEW,                                                                   \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                              \
   _GENERATE_SOA_CONST_VIEW_PART_1(CONST_VIEW, VIEW,                                                                   \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define GENERATE_SOA_CONST_VIEW(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                            \
   _GENERATE_SOA_CONST_VIEW(CONST_VIEW, BOOST_PP_CAT(CONST_VIEW, Unused_),                                             \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define _GENERATE_SOA_TRIVIAL_CONST_VIEW(CLASS, LAYOUTS_LIST, VALUE_LIST)                                              \
   _GENERATE_SOA_CONST_VIEW_PART_0_NO_DEFAULTS(ConstViewTemplateFreeParams, ViewTemplateFreeParams,                    \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                              \
   using BOOST_PP_CAT(CLASS, _parametrized) = CLASS<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT>;                       \
   _GENERATE_SOA_CONST_VIEW_PART_1(ConstViewTemplateFreeParams, ViewTemplateFreeParams,                                \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define _GENERATE_SOA_VIEW(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                                 \
   _GENERATE_SOA_VIEW_PART_0(CONST_VIEW, VIEW, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))    \
   _GENERATE_SOA_VIEW_PART_1(CONST_VIEW, VIEW, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))    \
   _GENERATE_SOA_VIEW_PART_2(CONST_VIEW, VIEW, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define GENERATE_SOA_VIEW(CONST_VIEW, VIEW, LAYOUTS_LIST, VALUE_LIST)                                                  \
   _GENERATE_SOA_CONST_VIEW(CONST_VIEW, VIEW, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))     \
   _GENERATE_SOA_VIEW(CONST_VIEW, VIEW, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define _GENERATE_SOA_TRIVIAL_VIEW(CLASS, LAYOUTS_LIST, VALUE_LIST, ...)                                               \
   _GENERATE_SOA_VIEW_PART_0_NO_DEFAULTS(ConstViewTemplateFreeParams, ViewTemplateFreeParams,                          \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                              \
   using BOOST_PP_CAT(CLASS, _parametrized) = CLASS<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT>;                       \
   _GENERATE_SOA_VIEW_PART_1(ConstViewTemplateFreeParams, ViewTemplateFreeParams,                                      \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                              \
                                                                                                                       \
   /* Extra operator=() for mutable element to emulate the aggregate initialisation syntax */                          \
   SOA_HOST_DEVICE SOA_INLINE constexpr element & operator=(const typename                                             \
       BOOST_PP_CAT(CLASS, _parametrized)::Metadata::value_element value) {                                            \
     _ITERATE_ON_ALL(_TRIVIAL_VIEW_ASSIGN_VALUE_ELEMENT, ~, __VA_ARGS__)                                               \
     return *this;                                                                                                     \
   }                                                                                                                   \
                                                                                                                       \
   _GENERATE_SOA_VIEW_PART_2(ConstViewTemplateFreeParams, ViewTemplateFreeParams,                                      \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))
// clang-format on

/**
 * Helper macro turning layout field declaration into view field declaration.
 */
#define _VIEW_FIELD_FROM_LAYOUT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, DATA) (DATA, NAME, NAME)

#define _VIEW_FIELD_FROM_LAYOUT(R, DATA, VALUE_TYPE_NAME) \
  BOOST_PP_EXPAND((_VIEW_FIELD_FROM_LAYOUT_IMPL BOOST_PP_TUPLE_PUSH_BACK(VALUE_TYPE_NAME, DATA)))

#endif  // DataFormats_SoATemplate_interface_SoAView_h
