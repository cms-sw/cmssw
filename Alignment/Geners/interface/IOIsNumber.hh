#ifndef GENERS_IOISNUMBER_HH_
#define GENERS_IOISNUMBER_HH_

namespace gs {
    template <class T>
    struct IOIsNumber
    {
        enum {value = 0};
    };
}

#define gs_declare_type_as_number(T) /**/                                \
namespace gs {                                                           \
    template <> struct IOIsNumber<T> {enum {value = 1};};                \
    template <> struct IOIsNumber<T const> {enum {value = 1};};          \
    template <> struct IOIsNumber<T volatile> {enum {value = 1};};       \
    template <> struct IOIsNumber<T const volatile> {enum {value = 1};}; \
}

gs_declare_type_as_number(float)
gs_declare_type_as_number(double)
gs_declare_type_as_number(long double)
gs_declare_type_as_number(int)
gs_declare_type_as_number(unsigned)
gs_declare_type_as_number(long)
gs_declare_type_as_number(long long)
gs_declare_type_as_number(unsigned long)
gs_declare_type_as_number(unsigned long long)
gs_declare_type_as_number(short)
gs_declare_type_as_number(unsigned short)
gs_declare_type_as_number(char)
gs_declare_type_as_number(unsigned char)
gs_declare_type_as_number(signed char)

#endif // GENERS_IOISNUMBER_HH_

