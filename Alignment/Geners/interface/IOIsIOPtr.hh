#ifndef GENERS_IOISIOPTR_HH_
#define GENERS_IOISIOPTR_HH_

#include "Alignment/Geners/interface/IOPtr.hh"

namespace gs {
    template <class T>
    struct IOIsIOPtr
    {
        enum {value = 0};
    };

    template <class T>
    struct IOIsIOPtr<IOPtr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<const IOPtr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<volatile IOPtr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<const volatile IOPtr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<IOProxy<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<const IOProxy<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<volatile IOProxy<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsIOPtr<const volatile IOProxy<T> >
    {
        enum {value = 1};
    };
}

#endif // GENERS_IOISIOPTR_HH_

