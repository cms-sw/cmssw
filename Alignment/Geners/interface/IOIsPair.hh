#ifndef GENERS_IOISPAIR_HH_
#define GENERS_IOISPAIR_HH_

#include <utility>

namespace gs {
    template <class T>
    struct IOIsPair
    {
        enum {value = 0};
    };

    template <class T1, class T2>
    struct IOIsPair<std::pair<T1,T2> >
    {
        enum {value = 1};
    };
}

#endif // GENERS_IOISPAIR_HH_

