/*
 * Structure-of-Arrays template allowing access to a selection of scalars and columns from one
 * or multiple SoAStores. A SoAStore is already a view to its complete set of columns.
 * This class will allow handling subsets of columns or set of columns from multiple SoAViews, possibly
 * with varying columns lengths.
 */

#ifndef DataStructures_SoAView_h
#define DataStructures_SoAView_h

#include "SoACommon.h"

#define SoA_view_store(TYPE, NAME) (TYPE, NAME)

#define SoA_view_store_list(...) __VA_ARGS__

#define SoA_view_value(STORE_NAME, STORE_MEMBER, LOCAL_NAME) (STORE_NAME, STORE_MEMBER, LOCAL_NAME)

#define SoA_view_value_list(...) __VA_ARGS__

/*
 * A macro defining a SoA view (collection of coluns from multiple stores)
 * 
 * Usage:
 * generate_SoA_view(PixelXYView,
 *   SoA_view_store_list(
 *     SoA_view_store(PixelDigis,        pixelDigis),
 *     SoA_view_store(PixelRecHitsStore, pixelsRecHit)
 *   ),
 *   SoA_view_value_list(
 *     SoA_view_value(pixelDigis,   x,   digisX),
 *     SoA_view_value(pixelDigis,   y,   digisY),
 *     SoA_view_value(pixelsRecHit, x, recHitsX),
 *     SoA_view_value(pixelsRecHit, y, recHitsY)
 *   )
 * );
 *    
 */

/*
 * Members definitions macros for viewa
 */

/**
 * Store types aliasing for referencing by name
 */
#define _DECLARE_VIEW_STORE_TYPE_ALIAS_IMPL(TYPE, NAME) typedef TYPE BOOST_PP_CAT(TypeOf_, NAME);

#define _DECLARE_VIEW_STORE_TYPE_ALIAS(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_STORE_TYPE_ALIAS_IMPL TYPE_NAME)

/**
 * Member types aliasing for referencing by name
 */
#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME, DATA)         \
  typedef BOOST_PP_CAT(TypeOf_, STORE_NAME)::SoAMetadata::BOOST_PP_CAT(TypeOf_, STORE_MEMBER)    \
      BOOST_PP_CAT(TypeOf_, LOCAL_NAME);                                                         \
  static const SoAColumnType BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME) =                           \
      BOOST_PP_CAT(TypeOf_, STORE_NAME)::SoAMetadata::BOOST_PP_CAT(ColumnTypeOf_, STORE_MEMBER); \
  SOA_HOST_DEVICE_INLINE                                                                         \
  DATA BOOST_PP_CAT(TypeOf_, LOCAL_NAME) * BOOST_PP_CAT(addressOf_, LOCAL_NAME)() const {        \
    return parent_.BOOST_PP_CAT(LOCAL_NAME, _);                                                  \
  };                                                                                             \
  static_assert(BOOST_PP_CAT(ColumnTypeOf_, LOCAL_NAME) != SoAColumnType::eigen,                 \
                "Eigen columns not supported in views.");

#define _DECLARE_VIEW_MEMBER_TYPE_ALIAS(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_TYPE_ALIAS_IMPL BOOST_PP_TUPLE_PUSH_BACK(STORE_MEMBER_NAME, DATA))

/**
 * Member assignment for trivial constructor
 */
#define _DECLARE_VIEW_MEMBER_TRIVIAL_CONSTRUCTION_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME) \
  (BOOST_PP_CAT(LOCAL_NAME, _)(nullptr))

#define _DECLARE_VIEW_MEMBER_TRIVIAL_CONSTRUCTION(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_TRIVIAL_CONSTRUCTION_IMPL STORE_MEMBER_NAME)

/**
 * Generator of parameters (stores) for constructor.
 */
#define _DECLARE_VIEW_CONSTRUCTION_PARAMETERS_IMPL(STORE_TYPE, STORE_NAME, DATA) (DATA STORE_TYPE & STORE_NAME)

#define _DECLARE_VIEW_CONSTRUCTION_PARAMETERS(R, DATA, TYPE_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS_IMPL BOOST_PP_TUPLE_PUSH_BACK(TYPE_NAME, DATA))

/**
 * Generator of member initialization from constructor.
 * We use a lambda with auto return type to handle multiple possible return types.
 */
#define _DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL(STORE, MEMBER, NAME)                       \
  (BOOST_PP_CAT(NAME, _)([&]() -> auto {                                                  \
    static_assert(BOOST_PP_CAT(SoAMetadata::ColumnTypeOf_, NAME) != SoAColumnType::eigen, \
                  "Eigen values not supported in views");                                 \
    return STORE.soaMetadata().BOOST_PP_CAT(addressOf_, MEMBER)();                        \
  }()))

#define _DECLARE_VIEW_MEMBER_INITIALIZERS(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_MEMBER_INITIALIZERS_IMPL STORE_MEMBER_NAME)

/**
 * Generator of element members initializer.
 */
#define _DECLARE_VIEW_ELEM_MEMBER_INIT_IMPL(STORE, MEMBER, LOCAL_NAME, DATA) (LOCAL_NAME(DATA, LOCAL_NAME))

#define _DECLARE_VIEW_ELEM_MEMBER_INIT(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEM_MEMBER_INIT_IMPL BOOST_PP_TUPLE_PUSH_BACK(STORE_MEMBER_NAME, DATA))

/**
 * Helper macro extracting the data type from a column accessor in a store
 */
#define _COLUMN_TYPE(STORE_NAME, STORE_MEMBER) \
  typename std::remove_pointer<decltype(BOOST_PP_CAT(STORE_NAME, Type)()::STORE_MEMBER())>::type

/**
 * Generator of parameters for (non-const) element subclass (expanded comma separated).
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME, DATA) \
  (DATA BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) * LOCAL_NAME)

#define _DECLARE_VIEW_ELEMENT_VALUE_ARG(R, DATA, STORE_MEMBER_NAME) \
  _DECLARE_VIEW_ELEMENT_VALUE_ARG_IMPL BOOST_PP_TUPLE_PUSH_BACK(STORE_MEMBER_NAME, DATA)

/**
 * Generator of member initialization for constructor of element subclass
 */
#define _DECLARE_VIEW_CONST_ELEM_MEMBER_INIT_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME, DATA) \
  (BOOST_PP_CAT(LOCAL_NAME, _)(DATA, LOCAL_NAME))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_VIEW_CONST_ELEM_MEMBER_INIT(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT_IMPL BOOST_PP_TUPLE_PUSH_BACK(STORE_MEMBER_NAME, DATA))

/**
 * Declaration of the members accessors of the const element subclass
 */
#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME)      \
  SOA_HOST_DEVICE_INLINE BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) LOCAL_NAME() const { \
    return BOOST_PP_CAT(LOCAL_NAME, _)();                                                    \
  }

#define _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR(R, DATA, STORE_MEMBER_NAME) \
  _DECLARE_VIEW_CONST_ELEMENT_ACCESSOR_IMPL STORE_MEMBER_NAME

/**
 * Declaration of the private members of the const element subclass
 */
#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME) \
  const SoAConstValue<BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)> BOOST_PP_CAT(LOCAL_NAME, _);

#define _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER(R, DATA, STORE_MEMBER_NAME) \
  _DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER_IMPL STORE_MEMBER_NAME

/**
 * Generator of the member-by-member copy operator of the element subclass.
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME) LOCAL_NAME() = other.LOCAL_NAME();

#define _DECLARE_VIEW_ELEMENT_VALUE_COPY(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_VALUE_COPY_IMPL STORE_MEMBER_NAME)

/**
 * Declaration of the private members of the const element subclass
 */
#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME) \
  SoAValue<BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME)> LOCAL_NAME;

#define _DECLARE_VIEW_ELEMENT_VALUE_MEMBER(R, DATA, STORE_MEMBER_NAME) \
  _DECLARE_VIEW_ELEMENT_VALUE_MEMBER_IMPL STORE_MEMBER_NAME

/**
 * Parameters passed to element subclass constructor in operator[]
 */
#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME) (BOOST_PP_CAT(LOCAL_NAME, _))

#define _DECLARE_VIEW_ELEMENT_CONSTR_CALL(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_ELEMENT_CONSTR_CALL_IMPL STORE_MEMBER_NAME)

/**
 * Direct access to column pointer and indexed access
 */
#define _DECLARE_VIEW_SOA_ACCESSOR_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME)                        \
  /* Column */                                                                                       \
  SOA_HOST_DEVICE_INLINE BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) * LOCAL_NAME() {             \
    return BOOST_PP_CAT(LOCAL_NAME, _);                                                              \
  }                                                                                                  \
  SOA_HOST_DEVICE_INLINE BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) & LOCAL_NAME(size_t index) { \
    return BOOST_PP_CAT(LOCAL_NAME, _)[index];                                                       \
  }

#define _DECLARE_VIEW_SOA_ACCESSOR(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_ACCESSOR_IMPL STORE_MEMBER_NAME)

/**
 * Direct access to column pointer (const) and indexed access.
 */
#define _DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME)                      \
  /* Column */                                                                                           \
  SOA_HOST_DEVICE_INLINE BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) const* LOCAL_NAME() const {      \
    return BOOST_PP_CAT(LOCAL_NAME, _);                                                                  \
  }                                                                                                      \
  SOA_HOST_DEVICE_INLINE BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) LOCAL_NAME(size_t index) const { \
    return *(BOOST_PP_CAT(LOCAL_NAME, _) + index);                                                       \
  }

#define _DECLARE_VIEW_SOA_CONST_ACCESSOR(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_CONST_ACCESSOR_IMPL STORE_MEMBER_NAME)

/**
 * SoA class member declaration (column pointers).
 */
#define _DECLARE_VIEW_SOA_MEMBER_IMPL(STORE_NAME, STORE_MEMBER, LOCAL_NAME, DATA) \
  DATA BOOST_PP_CAT(SoAMetadata::TypeOf_, LOCAL_NAME) * BOOST_PP_CAT(LOCAL_NAME, _) = nullptr;

#define _DECLARE_VIEW_SOA_MEMBER(R, DATA, STORE_MEMBER_NAME) \
  BOOST_PP_EXPAND(_DECLARE_VIEW_SOA_MEMBER_IMPL BOOST_PP_TUPLE_PUSH_BACK(STORE_MEMBER_NAME, DATA))

#define generate_SoA_view(CLASS, STORES_LIST, VALUE_LIST)                                                                               \
  struct CLASS {                                                                                                                        \
    /* these could be moved to an external type trait to free up the symbol names */                                                    \
    using self_type = CLASS;                                                                                                            \
                                                                                                                                        \
    /**                                                                                                                               \
   * Helper/friend class allowing SoA introspection.                                                                                \
   */ \
    struct SoAMetadata {                                                                                                                \
      friend CLASS;                                                                                                                     \
      /* Alias store types to name-derived identifyer to allow simpler definitions */                                                   \
      _ITERATE_ON_ALL(_DECLARE_VIEW_STORE_TYPE_ALIAS, ~, STORES_LIST)                                                                   \
                                                                                                                                        \
      /* Alias member types to name-derived identifyer to allow simpler definitions */                                                  \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, BOOST_PP_EMPTY(), VALUE_LIST)                                                    \
    private:                                                                                                                            \
      SOA_HOST_DEVICE_INLINE SoAMetadata(const CLASS& parent) : parent_(parent) {}                                                      \
      const CLASS& parent_;                                                                                                             \
    };                                                                                                                                  \
    friend SoAMetadata;                                                                                                                 \
    SOA_HOST_DEVICE_INLINE const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }                                         \
                                                                                                                                        \
    /* Trivial constuctor */                                                                                                            \
    CLASS() : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_TRIVIAL_CONSTRUCTION, ~, VALUE_LIST) {}                                        \
                                                                                                                                        \
    /* Constructor relying on user provided stores */                                                                                   \
    SOA_HOST_ONLY CLASS(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS, BOOST_PP_EMPTY(), STORES_LIST))                    \
        : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS, ~, VALUE_LIST) {}                                                    \
                                                                                                                                        \
    struct const_element {                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      const_element(size_t index, /* Declare parameters */                                                                              \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, const, VALUE_LIST))                                          \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                           \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_ACCESSOR, ~, VALUE_LIST)                                                              \
                                                                                                                                        \
    private:                                                                                                                            \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                                          \
    };                                                                                                                                  \
                                                                                                                                        \
    struct element {                                                                                                                    \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      element(size_t index, /* Declare parameters */                                                                                    \
              _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, BOOST_PP_EMPTY(), VALUE_LIST))                                     \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                                 \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      element& operator=(const element& other) {                                                                                        \
        _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_COPY, ~, VALUE_LIST)                                                                \
        return *this;                                                                                                                   \
      }                                                                                                                                 \
      _ITERATE_ON_ALL(_DECLARE_VIEW_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                                                \
    };                                                                                                                                  \
                                                                                                                                        \
    /* AoS-like accessor (non-const) */                                                                                                 \
    SOA_HOST_DEVICE_INLINE                                                                                                              \
    element operator[](size_t index) {                                                                                                  \
      return element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST));                                   \
    }                                                                                                                                   \
                                                                                                                                        \
    /* AoS-like accessor (const) */                                                                                                     \
    SOA_HOST_DEVICE_INLINE                                                                                                              \
    const const_element operator[](size_t index) const {                                                                                \
      return const_element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST));                             \
    }                                                                                                                                   \
                                                                                                                                        \
    /* accessors */                                                                                                                     \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_ACCESSOR, ~, VALUE_LIST)                                                                          \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_CONST_ACCESSOR, ~, VALUE_LIST)                                                                    \
                                                                                                                                        \
    /* dump the SoA internal structure */                                                                                               \
    template <typename T>                                                                                                               \
    SOA_HOST_ONLY friend void dump();                                                                                                   \
                                                                                                                                        \
  private:                                                                                                                              \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_MEMBER, BOOST_PP_EMPTY(), VALUE_LIST)                                                             \
  }

#define generate_SoA_const_view(CLASS, STORES_LIST, VALUE_LIST)                                                                         \
  struct CLASS {                                                                                                                        \
    /* these could be moved to an external type trait to free up the symbol names */                                                    \
    using self_type = CLASS;                                                                                                            \
                                                                                                                                        \
    /**                                                                                                                               \
   * Helper/friend class allowing SoA introspection.                                                                                \
   */ \
    struct SoAMetadata {                                                                                                                \
      friend CLASS;                                                                                                                     \
      /* Alias store types to name-derived identifyer to allow simpler definitions */                                                   \
      _ITERATE_ON_ALL(_DECLARE_VIEW_STORE_TYPE_ALIAS, ~, STORES_LIST)                                                                   \
                                                                                                                                        \
      /* Alias member types to name-derived identifyer to allow simpler definitions */                                                  \
      _ITERATE_ON_ALL(_DECLARE_VIEW_MEMBER_TYPE_ALIAS, const, VALUE_LIST)                                                               \
    private:                                                                                                                            \
      SOA_HOST_DEVICE_INLINE SoAMetadata(const CLASS& parent) : parent_(parent) {}                                                      \
      const CLASS& parent_;                                                                                                             \
    };                                                                                                                                  \
    friend SoAMetadata;                                                                                                                 \
    SOA_HOST_DEVICE_INLINE const SoAMetadata soaMetadata() const { return SoAMetadata(*this); }                                         \
                                                                                                                                        \
    /* Trivial constuctor */                                                                                                            \
    CLASS() : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_TRIVIAL_CONSTRUCTION, ~, VALUE_LIST) {}                                        \
                                                                                                                                        \
    /* Constructor relying on user provided stores */                                                                                   \
    SOA_HOST_ONLY CLASS(_ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONSTRUCTION_PARAMETERS, const, STORES_LIST))                               \
        : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_MEMBER_INITIALIZERS, ~, VALUE_LIST) {}                                                    \
                                                                                                                                        \
    struct const_element {                                                                                                              \
      SOA_HOST_DEVICE_INLINE                                                                                                            \
      const_element(size_t index, /* Declare parameters */                                                                              \
                    _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_VALUE_ARG, const, VALUE_LIST))                                          \
          : _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_CONST_ELEM_MEMBER_INIT, index, VALUE_LIST) {}                                           \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_ACCESSOR, ~, VALUE_LIST)                                                              \
                                                                                                                                        \
    private:                                                                                                                            \
      _ITERATE_ON_ALL(_DECLARE_VIEW_CONST_ELEMENT_VALUE_MEMBER, ~, VALUE_LIST)                                                          \
    };                                                                                                                                  \
                                                                                                                                        \
    /* AoS-like accessor (const) */                                                                                                     \
    SOA_HOST_DEVICE_INLINE                                                                                                              \
    const const_element operator[](size_t index) const {                                                                                \
      return const_element(index, _ITERATE_ON_ALL_COMMA(_DECLARE_VIEW_ELEMENT_CONSTR_CALL, ~, VALUE_LIST));                             \
    }                                                                                                                                   \
                                                                                                                                        \
    /* accessors */                                                                                                                     \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_CONST_ACCESSOR, ~, VALUE_LIST)                                                                    \
                                                                                                                                        \
    /* dump the SoA internal structure */                                                                                               \
    template <typename T>                                                                                                               \
    SOA_HOST_ONLY friend void dump();                                                                                                   \
                                                                                                                                        \
  private:                                                                                                                              \
    _ITERATE_ON_ALL(_DECLARE_VIEW_SOA_MEMBER, const, VALUE_LIST)                                                                        \
  }

#endif  // ndef DataStructures_SoAView_h
