#ifndef GENERS_CSTRINGBUF_HH_
#define GENERS_CSTRINGBUF_HH_

#include <sstream>

namespace gs {
    class CStringBuf : public std::stringbuf
    {
    public:
        explicit CStringBuf(std::ios_base::openmode mode = 
                            std::ios_base::in | std::ios_base::out)
            : std::stringbuf(mode) {}

        const char* getGetBuffer(unsigned long long* len) const;
        const char* getPutBuffer(unsigned long long* len) const;

    private:
        CStringBuf(const CStringBuf&);
        CStringBuf& operator=(const CStringBuf&);
    };
}

#endif // GENERS_CSTRINGBUF_HH_

