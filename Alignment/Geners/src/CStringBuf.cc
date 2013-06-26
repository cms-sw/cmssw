#include <cassert>

#include "Alignment/Geners/interface/CStringBuf.hh"

namespace gs {
    const char* CStringBuf::getGetBuffer(unsigned long long* len) const
    {
        const long long delta = gptr() - eback();
        assert(delta >= 0LL);
        assert(len);
        *len = delta;
        return eback();
    }

    const char* CStringBuf::getPutBuffer(unsigned long long* len) const
    {
        const long long delta = pptr() - pbase();
        assert(delta >= 0LL);
        assert(len);
        *len = delta;
        return pbase();
    }
}
