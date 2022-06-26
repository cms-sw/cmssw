/*
 * Structure-of-Arrays templates allowing access to a selection of scalars and columns from one
 * or multiple SoA layouts or views.
 * This template generator will allow handling subsets of columns from one or multiple SoA views or layouts.
 */

#ifndef DataStructures_SoAView_h
#define DataStructures_SoAView_h

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
 * GENERATE_SOA_VIEW(PixelXYView,
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
    SOA_HOST_DEVICE_INLINE ConstValueTraits(size_type, const typename C::valueType*) {}
    SOA_HOST_DEVICE_INLINE ConstValueTraits(size_type, const typename C::Params&) {}
    SOA_HOST_DEVICE_INLINE ConstValueTraits(size_type, const typename C::ConstParams&) {}
    // Any attempt to do anything with the "scalar" value a const element will fail.
  };

}  // namespace cms::soa

#include <memory_resource>
/*
 * Members definitions macros for viewa
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
#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA)                         \
  using BOOST_PP_CAT(TypeOf_, LOCAL_NAME) =                                                                        \
      typename BOOST_PP_CAT(TypeOf_, LAYOUT_NAME)::SoAMetadata::BOOST_PP_CAT(TypeOf_, LAYOUT_MEMBER);              \
  using BOOST_PP_CAT(ParametersTypeOf_, LOCAL_NAME) =                                                              \
      typename BOOST_PP_CAT(TypeOf_, LAYOUT_NAME)::SoAMetadata::BOOST_PP_CAT(ParametersTypeOf_, LAYOUT_MEMBER);    \
  constexpr static cms::soa::SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME) =                               \
      BOOST_PP_CAT(TypeOf_, LAYOUT_NAME)::SoAMetadata::BOOST_PP_CAT(ColumnTypeOf_, LAYOUT_MEMBER);                 \
  SOA_HOST_DEVICE_INLINE DATA auto* BOOST_PP_CAT(addressOf_, LOCAL_NAME)() const {                                 \
    return parent_.soaMetadata().BOOST_PP_CAT(parametersOf_, LOCAL_NAME)().addr_;                                  \
  };                                                                                                               \
  SOA_HOST_DEVICE_INLINE                                                                                           \
  DATA BOOST_PP_CAT(ParametersTypeOf_, LOCAL_NAME) BOOST_PP_CAT(parametersOf_, LOCAL_NAME)() const {               \
    return parent_.BOOST_PP_CAT(LOCAL_NAME, Parameters_);                                                          \
  };
// clang-format on

#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

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
  (DATA typename BOOST_PP_CAT(SoAMetadata::ParametersTypeOf_, LOCAL_NAME)::TupleOrPointerType LOCAL_NAME)

#define _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(                                                                  \
      _DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL(LAYOUT, MEMBER, NAME)                     \
  (BOOST_PP_CAT(NAME, Parameters_)([&]() -> auto {                                       \
    auto params = LAYOUT.soaMetadata().BOOST_PP_CAT(parametersOf_, MEMBER)();            \
    if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)                \
      if (reinterpret_cast<intptr_t>(params.addr_) % alignment)                          \
        throw std::runtime_error("In constructor by layout: misaligned column: " #NAME); \
    return params;                                                                       \
  }()))
// clang-format on

#define _DECLARE_VIEW_MEMBER_INITIALIZERS(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL LAYOUT_MEMBER_NAME)

/**
 * Generator of size computation for constructor.
 * This is the per-layout part of the lambda checking they all have the same size.
 */
// clang-format off
#define _UPDATE_SIZE_OF_VIEW_IMPL(LAYOUT_TYPE, LAYOUT_NAME)                                \
  if (set) {                                                                               \
    if (ret != LAYOUT_NAME.soaMetadata().size())                                           \
      throw std::runtime_error("In constructor by layout: different sizes from layouts."); \
  } else {                                                                                 \
    ret = LAYOUT_NAME.soaMetadata().size();                                                \
    set = true;                                                                            \
  }
// clang-format on

#define _UPDATE_SIZE_OF_VIEW(R, DATA, TYPE_NAME) BOOST_PP_EXPAND(_UPDATE_SIZE_OF_VIEW_IMPL TYPE_NAME)

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
// clang-format off
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN_IMPL(LAYOUT, MEMBER, NAME)                             \
  (                                                                                                       \
    BOOST_PP_CAT(NAME, Parameters_)([&]() -> auto {                                                       \
      if constexpr (alignmentEnforcement == AlignmentEnforcement::Enforced)                               \
        if (SoAMetadata:: BOOST_PP_CAT(ParametersTypeOf_, NAME)::checkAlignment(NAME, alignment))         \
          throw std::runtime_error("In constructor by column: misaligned column: " #NAME);                \
      return NAME;                                                                                        \
    }())                                                                                                  \
  )
// clang-format on

#define _DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN_IMPL LAYOUT_MEMBER_NAME)

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
  (DATA typename BOOST_PP_CAT(SoAMetadata::ParametersTypeOf_, LOCAL_NAME) LOCAL_NAME)

#define _DECLARE_VIEW_ELEMENT_VALUE_ARG(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA)

/**
 * Generator of parameters for (const) element subclass (expanded comma separated).
 */
#define _DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  (DATA typename BOOST_PP_CAT(SoAMetadata::ParametersTypeOf_, LOCAL_NAME)::ConstType LOCAL_NAME)

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
#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                 \
  SOA_HOST_DEVICE_INLINE                                                                                  \
      const typename SoAConstValueWithConf<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME),          \
      const typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>::RefToConst                          \
      LOCAL_NAME() const {                                                                                \
    return BOOST_PP_CAT(LOCAL_NAME, _)();                                                                 \
  }
// clang-format on

#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL LAYOUT_MEMBER_NAME

/**
 * Declaration of the private members of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                      \
  const cms::soa::ConstValueTraits<SoAConstValueWithConf<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME),     \
                                                         typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>, \
                                   BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME)>                           \
      BOOST_PP_CAT(LOCAL_NAME, _);
// clang-format on

#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL LAYOUT_MEMBER_NAME

/**
 * Generator of the member-by-member copy operator of the element subclass.
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                    \
  if constexpr (SoAMetadata::BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME) != cms::soa::SoAColumnType::scalar) \
    LOCAL_NAME() = other.LOCAL_NAME();

#define _DECLARE_VIEW_ELEMENT_VALUE_COPY(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL LAYOUT_MEMBER_NAME)

/**
 * Declaration of the private members of the const element subclass
 */
// clang-format off
#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME) \
  SoAValueWithConf<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME),                \
                   typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>             \
      LOCAL_NAME;
// clang-format on

#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL LAYOUT_MEMBER_NAME

/**
 * Parameters passed to element subclass constructor in operator[]
 */
#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME) \
  (BOOST_PP_CAT(LOCAL_NAME, Parameters_))

#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL LAYOUT_MEMBER_NAME)

/**
 * Direct access to column pointer and indexed access
 */
// clang-format off
#define _DECLARE_VIEW_SOA_ACCESSOR_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                         \
  /* Column or scalar */                                                                                \
  SOA_HOST_DEVICE_INLINE                                                                                \
  typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>::            \
        template ColumnType<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType< \
            cms::soa::SoAAccessType::mutableAccess>::NoParamReturnType                                  \
  LOCAL_NAME() {                                                                                        \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>::   \
        template ColumnType<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType< \
            cms::soa::SoAAccessType::mutableAccess>(BOOST_PP_CAT(LOCAL_NAME, Parameters_))();           \
  }                                                                                                     \
  SOA_HOST_DEVICE_INLINE auto& LOCAL_NAME(size_type index) {                                            \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>::   \
        template ColumnType<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType< \
            cms::soa::SoAAccessType::mutableAccess>(BOOST_PP_CAT(LOCAL_NAME, Parameters_))(index);      \
  }
// clang-format on

#define _DECLARE_VIEW_SOA_ACCESSOR(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_ACCESSOR_IMPL LAYOUT_MEMBER_NAME)

/**
 * Direct access to column pointer (const) and indexed access.
 */
// clang-format off
#define _DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME)                   \
  /* Column or scalar */                                                                                \
  SOA_HOST_DEVICE_INLINE auto LOCAL_NAME() const {                                                      \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>::   \
        template ColumnType<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType< \
            cms::soa::SoAAccessType::constAccess>(BOOST_PP_CAT(LOCAL_NAME, Parameters_))();             \
  }                                                                                                     \
  SOA_HOST_DEVICE_INLINE auto LOCAL_NAME(size_type index) const {                                       \
    return typename cms::soa::SoAAccessors<typename BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)>::   \
        template ColumnType<BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, LOCAL_NAME)>::template AccessType< \
            cms::soa::SoAAccessType::constAccess>(BOOST_PP_CAT(LOCAL_NAME, Parameters_))(index);        \
  }
// clang-format on

#define _DECLARE_VIEW_SOA_CONST_ACCESSOR(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL LAYOUT_MEMBER_NAME)

/**
 * SoA class member declaration (column pointers and parameters).
 */
#define _DECLARE_VIEW_SOA_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  typename BOOST_PP_CAT(SoAMetadata::ParametersTypeOf_, LOCAL_NAME) BOOST_PP_CAT(LOCAL_NAME, Parameters_);

#define _DECLARE_VIEW_SOA_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_MEMBER_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/**
 * Const SoA class member declaration (column pointers and parameters).
 */
#define _DECLARE_CONST_VIEW_SOA_MEMBER_IMPL(LAYOUT_NAME, LAYOUT_MEMBER, LOCAL_NAME, DATA) \
  typename BOOST_PP_CAT(SoAMetadata::ParametersTypeOf_, LOCAL_NAME)::ConstType BOOST_PP_CAT(LOCAL_NAME, Parameters_);

#define _DECLARE_CONST_VIEW_SOA_MEMBER(R, DATA, LAYOUT_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_CONST_VIEW_SOA_MEMBER_IMPL BOOST_PP_TUPLE_PUSH_BACK(LAYOUT_MEMBER_NAME, DATA))

/* ---- MUTABLE VIEW -------------------------------------------------------------------------------------------------------------------- */
// clang-format off
#define _GENERATE_SOA_VIEW_PART_0(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                          \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                                          \
            bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                                                \
            bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Default,                                                                     \
            bool RANGE_CHECKING = cms::soa::RangeChecking::Default>                                                                         \
  struct CLASS {                                                                                                                            \
    /* Declare the parametrized layouts as the default */                                                                                   \
    /*BOOST_PP_SEQ_CAT(_ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE, ~, LAYOUTS_LIST))   */                                   \
    /* these could be moved to an external type trait to free up the symbol names */                                                        \
    using self_type = CLASS;
// clang-format on

// clang-format off
#define _GENERATE_SOA_VIEW_PART_0_NO_DEFAULTS(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                          \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                                          \
            bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                                                \
            bool RESTRICT_QUALIFY,                                                                                                          \
            bool RANGE_CHECKING>                                                                                                            \
  struct CLASS {                                                                                                                            \
    /* Declare the parametrized layouts as the default */                                                                                   \
    /*BOOST_PP_SEQ_CAT(_ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_PARAMETRIZED_TEMPLATE, ~, LAYOUTS_LIST))   */                                   \
    /* these could be moved to an external type trait to free up the symbol names */                                                        \
    using self_type = CLASS;
// clang-format on

/**
 * Split of the const view definition where the parametrized template alias for the layout is defined for layout trivial view.
 */

// clang-format off
#define _GENERATE_SOA_VIEW_PART_1(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                          \
    using size_type = cms::soa::size_type;                                                                                                  \
    using byte_size_type = cms::soa::byte_size_type;                                                                                        \
    using AlignmentEnforcement = cms::soa::AlignmentEnforcement;                                                                            \
                                                                                                                                            \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                                 \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                             \
   * up to compute capability 8.X.                                                                                                          \
   */                                                                                                                                       \
    constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;                                                \
    constexpr static byte_size_type alignment = VIEW_ALIGNMENT;                                                                             \
    constexpr static bool alignmentEnforcement = VIEW_ALIGNMENT_ENFORCEMENT;                                                                \
    constexpr static byte_size_type conditionalAlignment =                                                                                  \
        alignmentEnforcement == AlignmentEnforcement::Enforced ? alignment : 0;                                                             \
    constexpr static bool restrictQualify = RESTRICT_QUALIFY;                                                                               \
    constexpr static bool rangeChecking = RANGE_CHECKING;                                                                                   \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                                               \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                                                 \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;                                     \
                                                                                                                                            \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                                                 \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;                           \
                                                                                                                                            \
    /**                                                                                                                                     \
   * Helper/friend class allowing SoA introspection.                                                                                        \
   */                                                                                                                                       \
    struct SoAMetadata {                                                                                                                    \
      friend CLASS;                                                                                                                         \
      SOA_HOST_DEVICE_INLINE size_type size() const { return parent_.nElements_; }                                                          \
      /* Alias layout or view types to name-derived identifyer to allow simpler definitions */                                              \
      _ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_TYPE_ALIAS, ~, LAYOUTS_LIST)                                                                     \
                                                                                                                                            \
      /* Alias member types to name-derived identifyer to allow simpler definitions */                                                      \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, BOOST_PP_EMPTY(), VALUE_LIST)                                                        \
                                                                                                                                            \
      /* Forbid copying to avoid const correctness evasion */                                                                               \
      SoAMetadata& operator=(const SoAMetadata&) = delete;                                                                                  \
      SoAMetadata(const SoAMetadata&) = delete;                                                                                             \
                                                                                                                                            \
    private:                                                                                                                                \
      SOA_HOST_DEVICE_INLINE SoAMetadata(const CLASS& parent) : parent_(parent) {}                                                          \
      const CLASS& parent_;                                                                                                                 \
    };                                                                                                                                      \
    friend SoAMetadata;                                                                                                                     \
    SOA_HOST_DEVICE_INLINE const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }                                             \
    SOA_HOST_DEVICE_INLINE SoAMetadata soaMetadata() { return SoAMetadata(*this); }                                                         \
                                                                                                                                            \
    /* Trivial constuctor */                                                                                                                \
    CLASS() {}                                                                                                                              \
                                                                                                                                            \
    /* Constructor relying on user provided layouts or views */                                                                             \
    SOA_HOST_ONLY CLASS(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS, BOOST_PP_EMPTY(), LAYOUTS_LIST))                       \
        : nElements_([&]() -> size_type {                                                                                                   \
            bool set = false;                                                                                                               \
            size_type ret = 0;                                                                                                              \
            _ITERATE_ON_ALL(_UPDATE_SIZE_OF_VIEW, BOOST_PP_EMPTY(), LAYOUTS_LIST)                                                           \
            return ret;                                                                                                                     \
          }()),                                                                                                                             \
          _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS, ~, VALUE_LIST) {}                                                        \
                                                                                                                                            \
    /* Constructor relying on individually provided column addresses */                                                                     \
    SOA_HOST_ONLY CLASS(size_type nElements,                                                                                                \
                        _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS,                                               \
                                              BOOST_PP_EMPTY(),                                                                             \
                                              VALUE_LIST))                                                                                  \
        : nElements_(nElements), _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN, ~, VALUE_LIST) {}                        \
                                                                                                                                            \
    struct const_element {                                                                                                                  \
      SOA_HOST_DEVICE_INLINE                                                                                                                \
      const_element(size_type index, /* Declare parameters */                                                                               \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, const, VALUE_LIST))                                              \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                               \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_ACCESSOR, ~, VALUE_LIST)                                                                  \
                                                                                                                                            \
    private:                                                                                                                                \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                                              \
    };                                                                                                                                      \
                                                                                                                                            \
    struct element {                                                                                                                        \
      SOA_HOST_DEVICE_INLINE                                                                                                                \
      element(size_type index, /* Declare parameters */                                                                                     \
              _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, BOOST_PP_EMPTY(), VALUE_LIST))                                         \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                                     \
      SOA_HOST_DEVICE_INLINE                                                                                                                \
      element& operator=(const element& other) {                                                                                            \
        _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_COPY, ~, VALUE_LIST)                                                                    \
        return *this;                                                                                                                       \
      }                                                                                                                                     \
      _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                                                    \
    };                                                                                                                                      \
                                                                                                                                            \
    /* AoS-like accessor (non-const) */                                                                                                     \
    SOA_HOST_DEVICE_INLINE                                                                                                                  \
    element operator[](size_type index) {                                                                                                   \
      if constexpr (rangeChecking == cms::soa::RangeChecking::Enabled) {                                                                    \
        if (index >= nElements_)                                                                                                            \
          SOA_THROW_OUT_OF_RANGE("Out of range index in " #CLASS "::operator[]")                                                            \
      }                                                                                                                                     \
      return element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST));                                       \
    }                                                                                                                                       \
                                                                                                                                            \
    /* AoS-like accessor (const) */                                                                                                         \
    SOA_HOST_DEVICE_INLINE                                                                                                                  \
    const_element operator[](size_type index) const {                                                                                       \
      if constexpr (rangeChecking == cms::soa::RangeChecking::Enabled) {                                                                    \
        if (index >= nElements_)                                                                                                            \
          SOA_THROW_OUT_OF_RANGE("Out of range index in " #CLASS "::operator[]")                                                            \
      }                                                                                                                                     \
      return const_element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST));                                 \
    }                                                                                                                                       \
                                                                                                                                            \
    /* accessors */                                                                                                                         \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_ACCESSOR, ~, VALUE_LIST)                                                                              \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_CONST_ACCESSOR, ~, VALUE_LIST)                                                                        \
                                                                                                                                            \
    /* dump the SoA internal structure */                                                                                                   \
    template <typename T>                                                                                                                   \
    SOA_HOST_ONLY friend void dump();                                                                                                       \
                                                                                                                                            \
  private:                                                                                                                                  \
    size_type nElements_ = 0;                                                                                                               \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_MEMBER, BOOST_PP_EMPTY(), VALUE_LIST)                                                                 \
  };
// clang-format on

/* ---- CONST VIEW --------------------------------------------------------------------------------------------------------------------- */
// clang-format off
#define _GENERATE_SOA_CONST_VIEW_PART_0(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                  \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT = cms::soa::CacheLineSize::defaultSize,                                                 \
            bool VIEW_ALIGNMENT_ENFORCEMENT = cms::soa::AlignmentEnforcement::Relaxed,                                                    \
            bool RESTRICT_QUALIFY = cms::soa::RestrictQualify::Enabled,                                                                   \
            bool RANGE_CHECKING = cms::soa::RangeChecking::Disabled>                                                                      \
  struct CLASS {                                                                                                                          \
    /* these could be moved to an external type trait to free up the symbol names */                                                      \
    using self_type = CLASS;
// clang-format on

// clang-format off
#define _GENERATE_SOA_CONST_VIEW_PART_0_NO_DEFAULTS(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                      \
  template <CMS_SOA_BYTE_SIZE_TYPE VIEW_ALIGNMENT,                                                                                        \
            bool VIEW_ALIGNMENT_ENFORCEMENT,                                                                                              \
            bool RESTRICT_QUALIFY,                                                                                                        \
            bool RANGE_CHECKING>                                                                                                          \
  struct CLASS {                                                                                                                          \
    /* these could be moved to an external type trait to free up the symbol names */                                                      \
    using self_type = CLASS;
// clang-format on

/**
 * Split of the const view definition where the parametrized template alias for the layout is defined for layout trivial view.
 */

// clang-format off
#define _GENERATE_SOA_CONST_VIEW_PART_1(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                  \
    typedef cms::soa::AlignmentEnforcement AlignmentEnforcement;                                                                          \
                                                                                                                                          \
    /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                               \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                           \
   * up to compute capability 8.X.                                                                                                        \
   */                                                                                                                                     \
    constexpr static byte_size_type defaultAlignment = cms::soa::CacheLineSize::defaultSize;                                              \
    constexpr static byte_size_type alignment = VIEW_ALIGNMENT;                                                                           \
    constexpr static bool alignmentEnforcement = VIEW_ALIGNMENT_ENFORCEMENT;                                                              \
    constexpr static byte_size_type conditionalAlignment =                                                                                \
        alignmentEnforcement == AlignmentEnforcement::Enforced ? alignment : 0;                                                           \
    constexpr static bool restrictQualify = RESTRICT_QUALIFY;                                                                             \
    constexpr static bool rangeChecking = RANGE_CHECKING;                                                                                 \
    /* Those typedefs avoid having commas in macros (which is problematic) */                                                             \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                                               \
    using SoAValueWithConf = cms::soa::SoAValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;                                   \
                                                                                                                                          \
    template <cms::soa::SoAColumnType COLUMN_TYPE, class C>                                                                               \
    using SoAConstValueWithConf = cms::soa::SoAConstValue<COLUMN_TYPE, C, conditionalAlignment, restrictQualify>;                         \
    /**                                                                                                                                   \
   * Helper/friend class allowing SoA introspection.                                                                                      \
   */                                                                                                                                     \
    struct SoAMetadata {                                                                                                                  \
      friend CLASS;                                                                                                                       \
      SOA_HOST_DEVICE_INLINE size_type size() const { return parent_.nElements_; }                                                        \
      /* Alias layout/view types to name-derived identifyer to allow simpler definitions */                                               \
      _ITERATE_ON_ALL(_DECLARE_VIEW_LAYOUT_TYPE_ALIAS, ~, LAYOUTS_LIST)                                                                   \
                                                                                                                                          \
      /* Alias member types to name-derived identifyer to allow simpler definitions */                                                    \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, const, VALUE_LIST)                                                                 \
                                                                                                                                          \
      SoAMetadata& operator=(const SoAMetadata&) = delete;                                                                                \
      SoAMetadata(const SoAMetadata&) = delete;                                                                                           \
                                                                                                                                          \
    private:                                                                                                                              \
      SOA_HOST_DEVICE_INLINE SoAMetadata(const CLASS& parent) : parent_(parent) {}                                                        \
      const CLASS& parent_;                                                                                                               \
    };                                                                                                                                    \
    friend SoAMetadata;                                                                                                                   \
    SOA_HOST_DEVICE_INLINE const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }                                           \
                                                                                                                                          \
    /* Trivial constuctor */                                                                                                              \
    CLASS() {}                                                                                                                            \
                                                                                                                                          \
    /* Constructor relying on user provided layouts or views */                                                                           \
    SOA_HOST_ONLY CLASS(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS, const, LAYOUTS_LIST))                                \
        : nElements_([&]() -> size_type {                                                                                                 \
            bool set = false;                                                                                                             \
            size_type ret = 0;                                                                                                            \
            _ITERATE_ON_ALL(_UPDATE_SIZE_OF_VIEW, BOOST_PP_EMPTY(), LAYOUTS_LIST)                                                         \
            return ret;                                                                                                                   \
          }()),                                                                                                                           \
          _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS, ~, VALUE_LIST) {}                                                      \
                                                                                                                                          \
    /* Constructor relying on individually provided column addresses */                                                                   \
    SOA_HOST_ONLY CLASS(size_type nElements,                                                                                              \
                        _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_BYCOLUMN_PARAMETERS, const, VALUE_LIST))                         \
        : nElements_(nElements), _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS_BYCOLUMN, ~, VALUE_LIST) {}                      \
                                                                                                                                          \
    struct const_element {                                                                                                                \
      SOA_HOST_DEVICE_INLINE                                                                                                              \
      const_element(size_type index, /* Declare parameters */                                                                             \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_CONST_VIEW_ELEMENT_VALUE_ARG, const, VALUE_LIST))                                      \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                             \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_ACCESSOR, ~, VALUE_LIST)                                                                \
                                                                                                                                          \
    private:                                                                                                                              \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                                            \
    };                                                                                                                                    \
                                                                                                                                          \
    /* AoS-like accessor (const) */                                                                                                       \
    SOA_HOST_DEVICE_INLINE                                                                                                                \
    const_element operator[](size_type index) const {                                                                                     \
      if constexpr (rangeChecking == cms::soa::RangeChecking::Enabled) {                                                                  \
        if (index >= nElements_)                                                                                                          \
          SOA_THROW_OUT_OF_RANGE("Out of range index in " #CLASS "::operator[]")                                                          \
      }                                                                                                                                   \
      return const_element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST));                               \
    }                                                                                                                                     \
                                                                                                                                          \
    /* accessors */                                                                                                                       \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_CONST_ACCESSOR, ~, VALUE_LIST)                                                                      \
                                                                                                                                          \
    /* dump the SoA internal structure */                                                                                                 \
    template <typename T>                                                                                                                 \
    SOA_HOST_ONLY friend void dump();                                                                                                     \
                                                                                                                                          \
  private:                                                                                                                                \
    size_type nElements_ = 0;                                                                                                             \
    _ITERATE_ON_ALL(_DECLARE_CONST_VIEW_SOA_MEMBER, const, VALUE_LIST)                                                                    \
};
// clang-format on

// clang-format off
// MAJOR caveat: in order to propagate the LAYOUTS_LIST and VALUE_LIST
#define GENERATE_SOA_VIEW(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                                \
   _GENERATE_SOA_VIEW_PART_0(CLASS, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                  \
   _GENERATE_SOA_VIEW_PART_1(CLASS, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define _GENERATE_SOA_TRIVIAL_VIEW(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                       \
   _GENERATE_SOA_VIEW_PART_0_NO_DEFAULTS(TrivialViewTemplateFreeParams,                                                                   \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                                                 \
   using  BOOST_PP_CAT(CLASS, _parametrized) = CLASS<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT>;                                         \
   _GENERATE_SOA_VIEW_PART_1(TrivialViewTemplateFreeParams, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define GENERATE_SOA_CONST_VIEW(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                          \
   _GENERATE_SOA_CONST_VIEW_PART_0(CLASS, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                            \
   _GENERATE_SOA_CONST_VIEW_PART_1(CLASS, SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))

#define _GENERATE_SOA_TRIVIAL_CONST_VIEW(CLASS, LAYOUTS_LIST, VALUE_LIST)                                                                 \
   _GENERATE_SOA_CONST_VIEW_PART_0_NO_DEFAULTS(TrivialConstViewTemplateFreeParams,                                                        \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))                                                                 \
   using  BOOST_PP_CAT(CLASS, _parametrized) = CLASS<VIEW_ALIGNMENT, VIEW_ALIGNMENT_ENFORCEMENT>;                                         \
   _GENERATE_SOA_CONST_VIEW_PART_1(TrivialConstViewTemplateFreeParams,                                                                    \
     SOA_VIEW_LAYOUT_LIST(LAYOUTS_LIST), SOA_VIEW_VALUE_LIST(VALUE_LIST))
// clang-format on
/**
 * Helper macro turning layout field declaration into view field declaration.
 */
#define _VIEW_FIELD_FROM_LAYOUT_IMPL(VALUE_TYPE, CPP_TYPE, NAME, DATA) (DATA, NAME, NAME)

#define _VIEW_FIELD_FROM_LAYOUT(R, DATA, VALUE_TYPE_NAME) \
  BOOST_PP_EXPAND((_VIEW_FIELD_FROM_LAYOUT_IMPL BOOST_PP_TUPLE_PUSH_BACK(VALUE_TYPE_NAME, DATA)))

/**
 * A macro defining both layout and view(s) in one go.
 */
// clang-format off
#define GENERATE_SOA_LAYOUT_VIEW_AND_CONST_VIEW(LAYOUT_NAME, VIEW_NAME, CONST_VIEW_NAME, ...)                          \
  GENERATE_SOA_LAYOUT(LAYOUT_NAME, __VA_ARGS__)                                                                        \
  using BOOST_PP_CAT(LAYOUT_NAME, _default) = LAYOUT_NAME<>;                                                           \
  GENERATE_SOA_VIEW(VIEW_NAME,                                                                                         \
    SOA_VIEW_LAYOUT_LIST((BOOST_PP_CAT(LAYOUT_NAME, _default), BOOST_PP_CAT(instance_, LAYOUT_NAME))),                 \
    SOA_VIEW_VALUE_LIST(_ITERATE_ON_ALL_COMMA(                                                                         \
    _VIEW_FIELD_FROM_LAYOUT, BOOST_PP_CAT(instance_, LAYOUT_NAME), __VA_ARGS__)))                                      \
  GENERATE_SOA_CONST_VIEW(                                                                                             \
    CONST_VIEW_NAME,                                                                                                   \
    SOA_VIEW_LAYOUT_LIST((BOOST_PP_CAT(LAYOUT_NAME, _default), BOOST_PP_CAT(instance_, LAYOUT_NAME))),                 \
    SOA_VIEW_VALUE_LIST(                                                                                               \
      _ITERATE_ON_ALL_COMMA(_VIEW_FIELD_FROM_LAYOUT, BOOST_PP_CAT(instance_, LAYOUT_NAME), __VA_ARGS__)))
// clang-format on

// clang-format off
#define GENERATE_SOA_LAYOUT_AND_CONST_VIEW(LAYOUT_NAME, CONST_VIEW_NAME, ...)                            \
  GENERATE_SOA_LAYOUT(LAYOUT_NAME, __VA_ARGS__)                                                          \
  using BOOST_PP_CAT(LAYOUT_NAME, _default) = LAYOUT_NAME<>;                                             \
  GENERATE_SOA_CONST_VIEW(                                                                               \
    CONST_VIEW_NAME,                                                                                     \
    SOA_VIEW_LAYOUT_LIST((BOOST_PP_CAT(LAYOUT_NAME, _default), BOOST_PP_CAT(instance_, LAYOUT_NAME))),   \
    SOA_VIEW_VALUE_LIST(                                                                                 \
      _ITERATE_ON_ALL_COMMA(_VIEW_FIELD_FROM_LAYOUT,                                                     \
        BOOST_PP_CAT(instance_, LAYOUT_NAME), __VA_ARGS__)))
// clang-format on

#endif  // ndef DataStructures_SoAView_h
