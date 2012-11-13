#ifndef GENERS_IOISCONTIGUOUS_HH_
#define GENERS_IOISCONTIGUOUS_HH_

#include <string>
#include <vector>

namespace gs {
    template <class T>
    struct IOIsContiguous
    {
        enum {value = 0};
    };

    // String is treated as a pod vector. This will be guaranteed
    // to work correctly in the C++11 standard. The current standard
    // does not specify that the characters must be stored contuguously
    // inside the string -- however, this is always true in practice.
    template <class T, class Traits, class Alloc>
    struct IOIsContiguous<std::basic_string<T,Traits,Alloc> >
    {enum {value = 1};};

    template <class T, class Traits, class Alloc>
    struct IOIsContiguous<const std::basic_string<T,Traits,Alloc> >
    {enum {value = 1};};

    template <class T, class Traits, class Alloc>
    struct IOIsContiguous<volatile std::basic_string<T,Traits,Alloc> >
    {enum {value = 1};};

    template <class T, class Traits, class Alloc>
    struct IOIsContiguous<const volatile std::basic_string<T,Traits,Alloc> >
    {enum {value = 1};};

    // std::vector is used by the package everywhere. No point in not
    // having it here.
    template <class T, class Alloc>
    struct IOIsContiguous<std::vector<T,Alloc> >
    {enum {value = 1};};

    template <class T, class Alloc>
    struct IOIsContiguous<const std::vector<T,Alloc> >
    {enum {value = 1};};

    template <class T, class Alloc>
    struct IOIsContiguous<volatile std::vector<T,Alloc> >
    {enum {value = 1};};

    template <class T, class Alloc>
    struct IOIsContiguous<const volatile std::vector<T,Alloc> >
    {enum {value = 1};};

    // Hovever, std::vector<bool> should be excluded
    template <>
    struct IOIsContiguous<std::vector<bool> >
    {enum {value = 0};};

    template <>
    struct IOIsContiguous<const std::vector<bool> >
    {enum {value = 0};};

    template <>
    struct IOIsContiguous<volatile std::vector<bool> >
    {enum {value = 0};};

    template <>
    struct IOIsContiguous<const volatile std::vector<bool> >
    {enum {value = 0};};
}

#endif // GENERS_IOISCONTIGUOUS_HH_

