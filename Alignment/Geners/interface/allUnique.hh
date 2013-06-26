// Checks whether all elements of a vector are unique

#ifndef GENERS_ALLUNIQUE_HH_
#define GENERS_ALLUNIQUE_HH_

#include <vector>

namespace gs {
    template<typename T>
    inline bool allUnique(const std::vector<T>& v)
    {
        const unsigned long sz = v.size();
        if (sz == 0UL)
            return true;
        const T* buf = &v[0];
        for (unsigned long i=1; i<sz; ++i)
            for (unsigned long j=0; j<i; ++j)
                if (buf[j] == buf[i])
                    return false;
        return true;
    }
}

#endif // GENERS_ALLUNIQUE_HH_

