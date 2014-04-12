#ifndef GENERS_FINDNAME_HH_
#define GENERS_FINDNAME_HH_

#include <vector>
#include <string>
#include <cassert>

namespace gs {
    inline unsigned long findName(const std::vector<std::string>& vec,
                                  const char* name)
    {
        assert(name);
        if (vec.empty())
            return 0UL;
        const std::string* names = &vec[0];
        const unsigned long ncols = vec.size();
        unsigned long col = 0;
        for (; col < ncols && names[col] != name; ++col) {;}
        return col;
    }
}

#endif // GENERS_FINDNAME_HH_

