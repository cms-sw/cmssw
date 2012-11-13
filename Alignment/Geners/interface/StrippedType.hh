// The code below strips away various qualifiers from a type

#ifndef GENERS_STRIPPEDTYPE_HH_
#define GENERS_STRIPPEDTYPE_HH_

#include <utility>

namespace gs {
    template <class T>
    struct StrippedType
    {
        typedef T type;
    };

    template <class T>
    struct StrippedType<T const>
    {
        typedef T type;
    };

    template <class T>
    struct StrippedType<T volatile>
    {
        typedef T type;
    };

    template <class T>
    struct StrippedType<T const volatile>
    {
        typedef T type;
    };

    template <class T, class U>
    struct StrippedType<std::pair<T,U> >
    {
        typedef std::pair<typename StrippedType<T>::type,
                          typename StrippedType<U>::type> type;
    };

    template <class T, class U>
    struct StrippedType<const std::pair<T,U> >
    {
        typedef std::pair<typename StrippedType<T>::type,
                          typename StrippedType<U>::type> type;
    };

    template <class T, class U>
    struct StrippedType<volatile std::pair<T,U> >
    {
        typedef std::pair<typename StrippedType<T>::type,
                          typename StrippedType<U>::type> type;
    };

    template <class T, class U>
    struct StrippedType<const volatile std::pair<T,U> >
    {
        typedef std::pair<typename StrippedType<T>::type,
                          typename StrippedType<U>::type> type;
    };
}

#endif // GENERS_STRIPPEDTYPE_HH_

