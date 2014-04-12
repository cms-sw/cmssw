#ifndef GENERS_IOPOINTEETYPE_HH_
#define GENERS_IOPOINTEETYPE_HH_

namespace gs {
    template <typename T>
    struct IOPointeeType;

    template <typename T>
    struct IOPointeeType<T*>
    {
        typedef T type;
    };

    template <typename T>
    struct IOPointeeType<T* const>
    {
        typedef T type;
    };

    template <typename T>
    struct IOPointeeType<T* volatile>
    {
        typedef T type;
    };

    template <typename T>
    struct IOPointeeType<T* const volatile>
    {
        typedef T type;
    };
}

#endif // GENERS_IOPOINTEETYPE_HH_

