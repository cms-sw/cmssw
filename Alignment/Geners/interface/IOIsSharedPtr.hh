#ifndef GENERS_IOISSHAREDPTR_HH_
#define GENERS_IOISSHAREDPTR_HH_

#include "Alignment/Geners/interface/CPP11_shared_ptr.hh"

namespace gs {
    template <class T>
    struct IOIsSharedPtr
    {
        enum {value = 0};
    };

    template <class T>
    struct IOIsSharedPtr<CPP11_shared_ptr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsSharedPtr<const CPP11_shared_ptr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsSharedPtr<volatile CPP11_shared_ptr<T> >
    {
        enum {value = 1};
    };

    template <class T>
    struct IOIsSharedPtr<const volatile CPP11_shared_ptr<T> >
    {
        enum {value = 1};
    };
}

#endif // GENERS_IOISSHAREDPTR_HH_

