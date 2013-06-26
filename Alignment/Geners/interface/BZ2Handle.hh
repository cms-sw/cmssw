#ifndef GENERS_BZ2HANDLE_HH_
#define GENERS_BZ2HANDLE_HH_

// There is no way to have a forward declaration of bz_stream
// because it is a typedef of an anonymous struct. Because of
// this, the bzlib header must be included here...
#include "bzlib.h"

// Note that, unlike similar Zlib handles, BZ2 handles manage
// external objects (typically living on the stack). This is
// because bz_stream has to be created every time a compression
// is performed, and having it on the stack saves a bit of time.

namespace gs {
    class BZ2InflateHandle
    {
    public:
        explicit BZ2InflateHandle(bz_stream& strm);
        ~BZ2InflateHandle();

    private:
        BZ2InflateHandle();
        BZ2InflateHandle(const BZ2InflateHandle&);
        BZ2InflateHandle& operator=(const BZ2InflateHandle&);

        bz_stream* strm_;
    };

    class BZ2DeflateHandle
    {
    public:
        explicit BZ2DeflateHandle(bz_stream& strm);
        ~BZ2DeflateHandle();

    private:
        BZ2DeflateHandle();
        BZ2DeflateHandle(const BZ2DeflateHandle&);
        BZ2DeflateHandle& operator=(const BZ2DeflateHandle&);

        bz_stream* strm_;
    };
}

#endif // GENERS_BZ2HANDLE_HH_

