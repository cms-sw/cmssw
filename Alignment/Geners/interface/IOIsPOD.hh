#ifndef GENERS_IOISPOD_HH_
#define GENERS_IOISPOD_HH_

#include "Alignment/Geners/interface/CPP11_type_traits.hh"

namespace gs {
    // In the following template, enum "IsPOD" is evaluated to 1
    // at compile time if T belongs to one of the known POD types.
    template <typename T>
    struct IOIsPOD
    {
        enum {value = CPP11_is_pod<T>::value};
    };
}

// Use the following macro (outside of any namespace)
// to declare some struct as POD for I/O purposes
//
#define gs_declare_type_as_pod(T) /**/                                \
namespace gs {                                                        \
    template <> struct IOIsPOD<T> {enum {value = 1};};                \
    template <> struct IOIsPOD<T const> {enum {value = 1};};          \
    template <> struct IOIsPOD<T volatile> {enum {value = 1};};       \
    template <> struct IOIsPOD<T const volatile> {enum {value = 1};}; \
}

#endif // GENERS_IOISPOD_HH_

