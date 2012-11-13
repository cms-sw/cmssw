#ifndef GENERS_IOISUNSIGNED_HH_
#define GENERS_IOISUNSIGNED_HH_

namespace gs {
    template <class T>
    struct IOIsUnsigned
    {
        enum {value = 0};
    };
}

#define gs_declare_type_as_unsigned(T) /**/                                \
namespace gs {                                                             \
    template <> struct IOIsUnsigned<T> {enum {value = 1};};                \
    template <> struct IOIsUnsigned<T const> {enum {value = 1};};          \
    template <> struct IOIsUnsigned<T volatile> {enum {value = 1};};       \
    template <> struct IOIsUnsigned<T const volatile> {enum {value = 1};}; \
}

gs_declare_type_as_unsigned(unsigned char)
gs_declare_type_as_unsigned(unsigned short)
gs_declare_type_as_unsigned(unsigned int)
gs_declare_type_as_unsigned(unsigned long)
gs_declare_type_as_unsigned(unsigned long long)

#endif // GENERS_IOISUNSIGNED_HH_

