#ifndef GENERS_IOISSTRING_HH_
#define GENERS_IOISSTRING_HH_

#include <string>

namespace gs {
    template <class T>
    struct IOIsString
    {
        enum {value = 0};
    };

    template <>
    struct IOIsString<std::string>
    {
        enum {value = 1};
    };

    template <>
    struct IOIsString<const std::string>
    {
        enum {value = 1};
    };

    template <>
    struct IOIsString<volatile std::string>
    {
        enum {value = 1};
    };

    template <>
    struct IOIsString<const volatile std::string>
    {
        enum {value = 1};
    };
}

#endif // GENERS_IOISSTRING_HH_

