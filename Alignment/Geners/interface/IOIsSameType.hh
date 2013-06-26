#ifndef GENERS_IOISSAMETYPE_HH_
#define GENERS_IOISSAMETYPE_HH_

namespace gs {
    template <typename T1, typename T2>
    struct IOIsSameType
    {
        enum {value = 0};
    };

    template <typename T>
    struct IOIsSameType<T, T>
    {
        enum {value = 1};
    };
}

#endif // GENERS_IOISSAMETYPE_HH_

